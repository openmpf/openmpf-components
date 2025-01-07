#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

from __future__ import annotations

import collections
import dataclasses
import importlib.resources
import itertools
import json
import logging
import typing
import uuid
from typing import (Any, Collection, Dict, Iterable, Iterator, List, Mapping,
                    NamedTuple, NewType, Optional, TextIO, Tuple, TypeVar)

import mpf_component_api as mpf
import mpf_component_util as mpf_util
import mpf_subject_api as mpf_sub
from mpf_subject_api import Multimap
from ortools.sat.python import cp_model

log = logging.getLogger(__name__)

FrameIndex = NewType('FrameIndex', int)

timing = mpf.Timing.with_logger(log)

class OrToolsSubjectComponent:

    @timing.time_func
    def get_subjects(self, job: mpf_sub.SubjectTrackingJob) -> mpf_sub.SubjectTrackingResults:
        log.info('Received subject tracking job.')
        media_id = get_media_id(job)

        config = get_config()
        min_iou = mpf_util.get_property(job.job_properties, 'MIN_IOU', 0.03)
        entity_type_to_tracks = get_tracks_grouped_by_entity_type(job, config)
        assignments = list(Assignment.assign_tracks(min_iou, entity_type_to_tracks))
        subject_entities = list(convert_subject_assignments(assignments))
        all_entities = {
            mpf_sub.EntityType('subject'): subject_entities
        }
        non_subject_entities = get_non_subject_entities(entity_type_to_tracks)
        for entity_type, entities in non_subject_entities.items():
            all_entities[entity_type] = [create_mpf_entity(e) for e in entities]

        relationships = get_relationship_dict(assignments, non_subject_entities, media_id)
        num_non_subject_entities = sum(len(es) for es in non_subject_entities.values())
        num_relationships = sum(len(rs) for rs in relationships.values())
        log.info(f'Finished running subject tracking job. Found {len(subject_entities)} subjects, '
                 f'{num_non_subject_entities} non-subject entities, '
                 f'and {num_relationships} relationships.')
        return mpf_sub.SubjectTrackingResults(all_entities, relationships)


@dataclasses.dataclass
class TrackWithTypes:
    id: mpf_sub.TrackId
    entity_type: mpf_sub.EntityType
    track_type: mpf_sub.TrackType
    track: mpf.VideoTrack
    frame_locations: Mapping[FrameIndex, mpf.ImageLocation] = dataclasses.field(init=False)
    detection_rects: Mapping[FrameIndex, mpf_util.Rect[int]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.frame_locations = typing.cast(
            Mapping[FrameIndex, mpf.ImageLocation],
            self.track.frame_locations)
        self.detection_rects = {
            fi: mpf_util.Rect.from_image_location(fl)
            for fi, fl in self.frame_locations.items()}

    def to_non_subject_entity(self) -> NonSubjectEntity:
        return NonSubjectEntity(uuid.uuid4(), self)


class NonSubjectEntity(NamedTuple):
    id: uuid.UUID
    track_info: TrackWithTypes


def get_config() -> TrackTypeConfig:
    with importlib.resources.open_text(__name__, 'config.json') as f:
        return TrackTypeConfig(f)

def get_media_id(job: mpf_sub.SubjectTrackingJob) -> mpf_sub.MediaId:
    if job.image_jobs or job.audio_jobs or job.generic_jobs:
        raise RuntimeError('Component only supports processing video detection jobs.')

    video_job_iter = iter(job.video_jobs)
    media_id = next(video_job_iter).media_id
    for video_job in video_job_iter:
        if media_id != video_job.media_id:
            raise RuntimeError('Component only supports jobs with a single piece of media.')
    return media_id


def get_tracks_grouped_by_entity_type(
        job: mpf_sub.SubjectTrackingJob,
        track_type_config: TrackTypeConfig) -> Multimap[mpf_sub.EntityType, TrackWithTypes]:
    entity_type_to_tracks = collections.defaultdict(list)
    for detection_job in job.video_jobs:
        component_type = detection_job.component_type
        for track_id, detection_track in detection_job.results.items():
            track_entity = convert_detection_track(
                    track_id, component_type, detection_track, track_type_config)
            entity_type_to_tracks[track_entity.entity_type].append(track_entity)
    return entity_type_to_tracks


def convert_detection_track(
            track_id: mpf_sub.TrackId,
            component_type: mpf_sub.DetectionComponentType,
            track: mpf.VideoTrack,
            track_type_config: TrackTypeConfig) -> TrackWithTypes:
    entity_type, track_type = track_type_config.get_entity_track_type(component_type, track)
    return TrackWithTypes(track_id, entity_type, track_type, track)


class TrackTypeConfig:
    def __init__(self, file: TextIO) -> None:
        config_json = json.load(file)
        self._track_type_props: Dict[str, List[str]] = {}
        for type_prop in config_json['track_type_properties']:
            type_ = type_prop['type'].lower()
            prop_names = [s.lower() for s in type_prop['properties']]
            self._track_type_props[type_] = prop_names

        self._entity_type_mappings: Dict[mpf_sub.TrackType, mpf_sub.EntityType] = {}
        for entity_mapping in config_json['entity_mappings']:
            entity_type = entity_mapping['entity_type'].lower()
            for track_type in entity_mapping['track_types']:
                self._entity_type_mappings[track_type.lower()] = entity_type

    def get_entity_track_type(
                self,
                component_type: mpf_sub.DetectionComponentType,
                track: mpf.VideoTrack) -> Tuple[mpf_sub.EntityType, mpf_sub.TrackType]:
        track_type = self._get_track_type(track, component_type)
        entity_type = self._entity_type_mappings.get(track_type)
        if entity_type is None:
            return mpf_sub.EntityType(track_type), track_type
        else:
            return entity_type, track_type

    def _get_track_type(
                self,
                track: mpf.VideoTrack,
                detection_component_type: str) -> mpf_sub.TrackType:
        detection_component_type = detection_component_type.lower()
        type_props = self._track_type_props.get(detection_component_type.lower(), ())
        track_props = {k.lower(): v for k, v in track.detection_properties.items()}
        for type_prop in type_props:
            try:
                return  mpf_sub.TrackType(track_props[type_prop.lower()])
            except KeyError:
                pass
        return  mpf_sub.TrackType(detection_component_type)


class Assignment:
    @classmethod
    @timing.time_func
    def assign_tracks(
            cls,
            min_iou: float,
            entity_type_to_tracks: Multimap[mpf_sub.EntityType, TrackWithTypes]
            ) -> Iterator[TrackAssignment]:
        log.info('Starting track assignment.')
        face_tracks = entity_type_to_tracks.get(mpf_sub.EntityType('face'), ())
        person_tracks = entity_type_to_tracks.get(mpf_sub.EntityType('person'), ())
        model = cp_model.CpModel()
        model_vars_grouped_by_person = collections.defaultdict(list)
        assignments_grouped_by_face = []
        for ft in face_tracks:
            face_to_person_assignments = []
            for pt in person_tracks:
                score = iou_sum(ft, pt, min_iou)
                if score > 0:
                    model_var = model.NewBoolVar('')
                    assignment_var = AssignmentVar(model_var, ft, pt, score)
                    face_to_person_assignments.append(assignment_var)
                    model_vars_grouped_by_person[id(pt)].append(model_var)
            if face_to_person_assignments:
                assignments_grouped_by_face.append(face_to_person_assignments)
            else:
                yield TrackAssignment.face_only(ft)

        for person_face_assignment_group in model_vars_grouped_by_person.values():
            # Prevent a face track from being assigned to more than one person track.
            model.AddAtMostOne(person_face_assignment_group)

        cls._add_overlap_constraints(assignments_grouped_by_face, model)

        objective_terms = (a.model_var * a.score for fg in assignments_grouped_by_face for a in fg)
        model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        status: Any = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ArithmeticError('No solution found.')

        yield from cls._create_output_assignments(
                person_tracks, assignments_grouped_by_face, solver)
        log.info('Track assignment completed.')


    @staticmethod
    def _add_overlap_constraints(
            assignment_vars: List[List[AssignmentVar]],
            model: cp_model.CpModel) -> None:
        # Ensure that none of the person tracks assigned to a subject have frames in common.
        for face_group in assignment_vars:
            for p1_assignment_var, p2_assignment_var in itertools.combinations(face_group, 2):
                if frames_in_common(p1_assignment_var.person_track, p2_assignment_var.person_track):
                    # The two person tracks have frames in common, so only one of the two can be
                    # assigned to a subject.
                    model.AddAtMostOne((p1_assignment_var.model_var, p2_assignment_var.model_var))


    @staticmethod
    def _create_output_assignments(
            person_tracks: Iterable[TrackWithTypes],
            assignment_vars: List[List[AssignmentVar]],
            solver: cp_model.CpSolver) -> Iterator[TrackAssignment]:
        all_assigned_person_ids = set()
        for face_group in assignment_vars:
            total_score = 0
            assigned_people = []
            for assignment in face_group:
                if solver.BooleanValue(assignment.model_var):
                    total_score += assignment.score
                    assigned_people.append(assignment.person_track)
                    all_assigned_person_ids.add(id(assignment.person_track))
            face_track = face_group[0].face_track
            if assigned_people:
                yield TrackAssignment(total_score, face_track, assigned_people)
            else:
                yield TrackAssignment.face_only(face_track)
        yield from (TrackAssignment.person_only(pt)
                    for pt in person_tracks if id(pt) not in all_assigned_person_ids)


class AssignmentVar(NamedTuple):
    model_var: cp_model.IntVar
    face_track: TrackWithTypes
    person_track: TrackWithTypes
    score: float


@dataclasses.dataclass
class TrackAssignment:
    score: float
    face_track: Optional[TrackWithTypes]
    person_tracks: Collection[TrackWithTypes]
    subject_id: uuid.UUID = dataclasses.field(init=False, default_factory=uuid.uuid4)
    detection_rects: Mapping[FrameIndex, List[mpf_util.Rect[int]]] = (
            dataclasses.field(init=False))

    def __post_init__(self):
        self.detection_rects = collections.defaultdict(list)
        if self.face_track:
            tracks = itertools.chain((self.face_track,), self.person_tracks)
        else:
            tracks = self.person_tracks
        for track in tracks:
            for frame_idx, rect in track.detection_rects.items():
                self.detection_rects[frame_idx].append(rect)


    @staticmethod
    def face_only(face_track: TrackWithTypes):
        return TrackAssignment(-1, face_track, ())

    @staticmethod
    def person_only(person_track: TrackWithTypes):
        return TrackAssignment(-1, None, (person_track,))


    def get_overlap_frames(self, other: TrackAssignment) -> List[FrameIndex]:
        return [
            fi for fi, rs1, rs2
                in get_items_with_common_keys(self.detection_rects, other.detection_rects)
            if any_intersection(rs1, rs2)]


def any_intersection(
        rects1: Iterable[mpf_util.Rect[int]],
        rects2: Iterable[mpf_util.Rect[int]]) -> bool:
    return any(rects_intersect(r1, r2) for r1 in rects1 for r2 in rects2)


def rects_intersect(r1: mpf_util.Rect[int], r2: mpf_util.Rect[int]) -> bool:
    left_rect, right_rect = sorted((r1, r2), key=lambda r: r.x)
    if right_rect.x > left_rect.x + left_rect.width:
        return False
    top_rect, bottom_rect = sorted((r1, r2), key=lambda r: r.y)
    return bottom_rect.y <= top_rect.y + top_rect.height


def iou_sum(track1: TrackWithTypes, track2: TrackWithTypes, min_iou: float) -> float:
    if not frame_ranges_overlap(track1, track2):
        return -1
    return sum(
        iou(r1, r2, min_iou)
        for _, r1, r2 in get_items_with_common_keys(track1.detection_rects, track2.detection_rects))


def iou(rect1: mpf_util.Rect[int], rect2: mpf_util.Rect[int], min_iou: float) -> float:
    if intersection := rect1 & rect2:
        result = intersection.area / (rect1.area + rect2.area - intersection.area)
        return result if result >= min_iou else 0
    else:
        return -1


def frame_ranges_overlap(track1: TrackWithTypes, track2: TrackWithTypes) -> bool:
    if track1.track.start_frame < track2.track.start_frame:
        return track1.track.stop_frame >= track2.track.start_frame
    else:
        return track2.track.stop_frame >= track1.track.start_frame


def frames_in_common(track1: TrackWithTypes, track2: TrackWithTypes) -> bool:
    if not frame_ranges_overlap(track1, track2):
        return False
    return any(get_items_with_common_keys(track1.frame_locations, track2.frame_locations))


def convert_subject_assignments(
            track_assignments: Iterable[TrackAssignment]) -> Iterable[mpf_sub.Entity]:
    for track_assignment in track_assignments:
        face_tracks = (track_assignment.face_track.id,) if track_assignment.face_track else ()
        person_track_ids = [p.id for p in track_assignment.person_tracks]
        grouped_track_ids = {
            mpf_sub.TrackType('face'): face_tracks,
            mpf_sub.TrackType('person'): person_track_ids
        }
        yield mpf_sub.Entity(
                track_assignment.subject_id, track_assignment.score, grouped_track_ids)



SUBJECT_ENTITY_TYPES = {mpf_sub.EntityType('face'), mpf_sub.EntityType('person')}

def get_non_subject_entities(grouped_tracks: Multimap[mpf_sub.EntityType, TrackWithTypes]
                             ) -> Multimap[mpf_sub.EntityType, NonSubjectEntity]:
    entities = collections.defaultdict(list)
    for entity_type, tracks in grouped_tracks.items():
        if entity_type not in SUBJECT_ENTITY_TYPES:
            entities[entity_type] = [t.to_non_subject_entity() for t in tracks]
    return entities


def create_mpf_entity(entity: NonSubjectEntity) -> mpf_sub.Entity:
    return mpf_sub.Entity(
            entity.id, -1,
            {entity.track_info.track_type: (entity.track_info.id,)})


@timing.time_func('Finding relationships')
def get_relationship_dict(
        assignments: Iterable[TrackAssignment],
        entity_groups: Multimap[mpf_sub.EntityType, NonSubjectEntity],
        media_id: mpf_sub.MediaId) -> Multimap[mpf_sub.RelationshipType, mpf_sub.Relationship]:
    relationships = itertools.chain(
            get_subject_to_entity_proximity_relationships(assignments, entity_groups, media_id),
            get_entity_proximity_relationships(entity_groups, media_id),
            get_subject_to_subject_proximity_relationships(assignments, media_id))
    return {mpf_sub.RelationshipType('proximity'): list(relationships)}


def get_subject_to_entity_proximity_relationships(
        assignments: Iterable[TrackAssignment],
        entity_groups: Multimap[mpf_sub.EntityType, NonSubjectEntity],
        media_id: mpf_sub.MediaId) -> Iterable[mpf_sub.Relationship]:
    for subject_track_assignment in assignments:
        flattened_entities = (e for eg in entity_groups.values() for e in eg)
        for entity in flattened_entities:
            if proximity_frames := get_subject_proximity_frames(subject_track_assignment, entity):
                ids = [subject_track_assignment.subject_id, entity.id]
                media_ref = mpf_sub.MediaReference(media_id, proximity_frames)
                yield mpf_sub.Relationship(ids, (media_ref,))


def get_subject_to_subject_proximity_relationships(
            assignments: Iterable[TrackAssignment],
            media_id: mpf_sub.MediaId) -> Iterable[mpf_sub.Relationship]:
    for assignment1, assignment2 in itertools.combinations(assignments, 2):
        if frames := assignment1.get_overlap_frames(assignment2):
            media_ref = mpf_sub.MediaReference(media_id, frames)
            ids = (assignment1.subject_id, assignment2.subject_id)
            yield mpf_sub.Relationship(ids, (media_ref,))


def get_entity_proximity_relationships(
        entity_groups: Multimap[mpf_sub.EntityType, NonSubjectEntity],
        media_id: mpf_sub.MediaId) -> Iterable[mpf_sub.Relationship]:

    for entity1, entity2 in get_entity_pairs(entity_groups):
        if frames := get_entity_proximity_frames(entity1, entity2):
            ids = (entity1.id, entity2.id)
            media_ref = mpf_sub.MediaReference(media_id, frames)
            yield mpf_sub.Relationship(ids, (media_ref,))

def get_subject_proximity_frames(
        subject: TrackAssignment,
        entity_track: NonSubjectEntity) -> List[FrameIndex]:
    return sorted(
        fi for fi, subj_rects, det_rect in get_items_with_common_keys(
                subject.detection_rects, entity_track.track_info.detection_rects)
        if any_intersection((det_rect,), subj_rects))



def get_entity_pairs(
        entity_groups: Multimap[mpf_sub.EntityType, NonSubjectEntity]
        ) -> Iterable[Tuple[NonSubjectEntity, NonSubjectEntity]]:
    for entity_group1, entity_group2 in itertools.combinations(entity_groups.values(), 2):
        yield from itertools.product(entity_group1, entity_group2)


def get_entity_proximity_frames(entity1: NonSubjectEntity, entity2: NonSubjectEntity) -> List[int]:
    return sorted(
        fi for fi, r1, r2 in get_items_with_common_keys(
                entity1.track_info.detection_rects, entity2.track_info.detection_rects)
        if rects_intersect(r1, r2))


K = TypeVar('K')
V1 = TypeVar('V1')
V2 = TypeVar('V2')

def get_items_with_common_keys(
        mapping1: Mapping[K, V1],
        mapping2: Mapping[K, V2]) -> Iterator[Tuple[K, V1, V2]]:
    if len(mapping1) <= len(mapping2):
        return ((k, v1, v2) for k, v1 in mapping1.items() if (v2 := mapping2.get(k)))
    else:
        return ((k, v1, v2) for k, v2 in mapping2.items() if (v1 := mapping1.get(k)))
