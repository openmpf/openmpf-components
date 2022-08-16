#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
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

import os
import uuid
from typing import (
    Optional, List, Iterable, Tuple, Mapping, NamedTuple, Any
)
from itertools import accumulate
from bisect import bisect_right

import mpf_component_api as mpf
import mpf_component_util as util

from .azure_connect import AzureConnection, AcsServerInfo


class SpeakerInfo(NamedTuple):
    speaker_id: str
    gender: str
    gender_score: float
    language: str
    language_scores: Mapping[str, float]
    speech_segs: List[Tuple[float, float]]


class AcsSpeechDetectionProcessor(object):

    def __init__(self, logger):
        self.logger = logger
        self.acs = AzureConnection(logger)

    @staticmethod
    def parse_segments_str(
                segments_string: str,
                media_start_time: Optional[int] = 0,
                media_stop_time: Optional[int] = None
            ) -> Optional[List[Tuple[int, int]]]:
        """ Converts a string of the form
                'start_1-stop_1, start_2-stop_2, ..., start_n-stop_n'
            where start_x and stop_x are in milliseconds, Olive-compatible
            AnnotationRegion object list.
        """
        try:
            segments = []
            start_stops = segments_string.split(",")
            for ss in start_stops:
                start, stop = ss.strip().split("-")
                start = int(start)
                stop = int(stop)

                # Limit to media start and stop times. If entirely
                #  outside the limits, drop the segment
                if media_stop_time is not None and start > media_stop_time:
                    continue
                if stop < media_start_time:
                    continue
                start = max(start, media_start_time)
                if media_stop_time is not None:
                    stop = min(stop, media_stop_time)

                # Offset by media_start_time so that segments are
                #  relative to the transcoded waveform
                start -= media_start_time
                stop -= media_start_time

                segments.append((start, stop))
        except Exception as e:
            raise mpf.DetectionException(
                'Exception raised while parsing voiced segments '
                f'"{segments_string}": {e}',
                mpf.DetectionError.INVALID_PROPERTY
            )

        # If all the voiced segments are outside the time range, signal that
        #  we should halt and return an empty list.
        if not segments:
            return None

        return sorted(segments)

    @staticmethod
    def desegment_times(
                utterances: Iterable[Mapping[str, Any]],
                speaker: Optional[SpeakerInfo] = None
            ) -> Iterable[Tuple[int, int, Iterable[Tuple[int, int]]]]:
        """ Adjust region start and stop times to account for the passed
            annotation_regions.

            :param utterances: A list of utterances returned by ACS
            :param speaker: SpeakerInfo. None of no feed-forward track exists
            :returns: An iterable of utterance starts, stops, and word segments.
        """
        if speaker is not None:
            # Create parallel lists for the beginning, end, and length of each
            #  feedforward segment, so that utterance timestamps can be shifted
            #  to be relative to the beginning of the input audio file.
            segbegs, seglens = zip(*(
                (t0, t1-t0) for t0, t1 in sorted(speaker.speech_segs)
            ))

            # The dividing line between each segment relative to regions
            segdivs = list(accumulate(seglens))

            # The offset to add to region times, according to their segment
            offsets = [a-b for a, b in zip(segbegs, [0] + segdivs)]

        utterance_times = []
        for utt in utterances:
            # Timing information. Azure works in 100-nanosecond ticks,
            #  so multiply by 1e-4 to obtain milliseconds.
            utterance_start = utt['offsetInTicks'] * 1e-4
            utterance_duration = utt['durationInTicks'] * 1e-4
            utterance_stop = utterance_start + utterance_duration
            word_starts, word_ends = zip(*[
                (
                    w['offsetInTicks'] * 1e-4,
                    (w['offsetInTicks'] + w['durationInTicks']) * 1e-4
                ) for w in utt['nBest'][0]['words']
            ])

            if speaker is not None:
                # Get index of the segment containing the region start time,
                #  and offset utterance time information by appropriate delta
                i = min(bisect_right(segdivs, utterance_start), len(segdivs)-1)
                utterance_start += offsets[i]
                utterance_stop += offsets[i]
                word_starts = [t + offsets[i] for t in word_starts]
                word_ends = [t + offsets[i] for t in word_ends]

            yield utterance_start, utterance_stop, zip(word_starts, word_ends)

    def process_audio(
                self, target_file: str, start_time: int,
                stop_time: Optional[int], job_name: str,
                server_info: AcsServerInfo, language: str, diarize: bool,
                cleanup: bool, blob_access_time: int, expiry: int,
                speaker: Optional[SpeakerInfo] = None
            ) -> List[mpf.AudioTrack]:
        self.acs.update_acs(server_info)

        self.logger.debug('Loading file: ' + target_file)
        audio_bytes = util.transcode_to_wav(
            target_file,
            start_time=start_time,
            stop_time=stop_time,
            segments=speaker.speech_segs if speaker else None
        )

        try:
            self.logger.info('Uploading file to blob')
            recording_id = str(uuid.uuid4())
            recording_url = self.acs.upload_file_to_blob(
                audio_bytes=audio_bytes,
                recording_id=recording_id,
                blob_access_time=blob_access_time
            )
        except mpf.DetectionException:
            raise
        except Exception as e:
            raise mpf.DetectionException(
                'Failed to upload file to blob: {}'.format(e),
                mpf.DetectionError.DETECTION_FAILED
            )

        output_loc = None
        while output_loc is None:
            try:
                self.logger.info('Submitting speech-to-text job to ACS')
                output_loc = self.acs.submit_batch_transcription(
                    recording_url=recording_url,
                    job_name=job_name,
                    diarize=diarize,
                    language=language,
                    expiry=expiry
                )
            except Exception as e:
                if 'This locale does not support diarization' in str(e):
                    self.logger.warning(
                        f'Locale "{lang}" does not support diarization. '
                        'Completing job with diarization disabled.')
                    diarize = False
                else:
                    if cleanup:
                        self.logger.info('Marking file blob for deletion')
                        self.acs.delete_blob(recording_id)
                    raise

        try:
            self.logger.info('Retrieving transcription')
            result = self.acs.poll_for_result(output_loc)

            if result['status'] == 'Failed':
                raise mpf.DetectionException(
                    'Transcription failed: {}'.format(result['statusMessage']),
                    mpf.DetectionError.DETECTION_FAILED
                )

            transcription = self.acs.get_transcription(result)
            self.logger.info('Speech-to-text processing complete')
        finally:
            if cleanup:
                self.logger.info('Marking file blob for deletion')
                self.acs.delete_blob(recording_id)
            self.logger.info('Deleting transcript')
            self.acs.delete_transcription(output_loc)

        utterances = transcription['recognizedPhrases']
        desegmented_times = self.desegment_times(
            utterances=utterances,
            speaker=speaker)

        self.logger.info('Completed process audio')
        self.logger.info('Creating AudioTracks')
        audio_tracks = []
        for utt, times in zip(utterances, desegmented_times):
            # Speaker ID returned by Azure diarization, default 0
            speaker_id = utt.get('speaker', '0')

            # Transcript text
            display = utt['nBest'][0]['display']

            # Timing information, desegmented if feed-forward track exists
            utterance_start, utterance_stop, word_segments = times

            # Utterance confidence (seemingly not an aggregate of word confidences)
            utterance_confidence = utt['nBest'][0]['confidence']

            # Build word confidence and segment strings
            word_confidences = ', '.join([
                str(w['confidence'])
                for w in utt['nBest'][0]['words']
            ])
            word_segments = ', '.join([
                f"{t0:f}-{t1:f}"
                for t0, t1 in word_segments
            ])

            properties = dict(
                SPEAKER_ID=speaker_id,
                TRANSCRIPT=display,
                WORD_CONFIDENCES=word_confidences,
                DECODED_LANGUAGE=language,
                WORD_SEGMENTS=word_segments
            )

            if speaker is not None:
                # If feed-forward track exists, use provided speaker info
                ldict = speaker.language_scores
                languages = [n for n, s in ldict.items()]
                languages = sorted(
                    languages,
                    key=lambda lab: ldict.get(lab, -1.0),
                    reverse=True
                )
                language_confs = ', '.join(str(ldict[lab]) for lab in languages)
                language_labels = ', '.join(languages)

                # Add or update properties according to fed-forward info
                properties.update(
                    SPEAKER_ID=speaker.speaker_id,
                    GENDER=speaker.gender,
                    GENDER_CONFIDENCE=speaker.gender_score,
                    SPEAKER_LANGUAGES=language_labels,
                    SPEAKER_LANGUAGE_CONFIDENCES=language_confs
                )

            track = mpf.AudioTrack(
                start_time=utterance_start,
                stop_time=utterance_stop,
                confidence=utterance_confidence,
                detection_properties=properties
            )
            audio_tracks.append(track)

        self.logger.info('Completed processing transcription results')
        return audio_tracks
