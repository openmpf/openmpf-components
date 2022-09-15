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

import logging
import uuid
from typing import (
    Optional, List, Iterable, Tuple, Mapping, Any, NamedTuple
)
from itertools import accumulate, dropwhile, takewhile, tee
from bisect import bisect_right
from collections import namedtuple

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from acs_speech_component.azure_connect import AzureConnection
from acs_speech_component.job_parsing import AzureJobConfig
from acs_speech_component.azure_utils import ISO6393_TO_BCP47


BCP47_TO_ISO6393 = {bcp: iso for iso, bcps in ISO6393_TO_BCP47.items() for bcp in bcps}


logger = logging.getLogger('AcsSpeechComponent')


class Utterance(NamedTuple):
    display: str
    segment: Tuple[int, int]
    confidence: float
    word_segments: List[Tuple[int, int]]
    word_confidences: List[float]
    speaker_id: str


class AcsSpeechDetectionProcessor(object):

    def __init__(self):
        self.acs = AzureConnection()

    @staticmethod
    def convert_word_timing(
                recognized_phrases: Iterable[Mapping[str, Any]],
                speaker: Optional[mpf_util.SpeakerInfo] = None
            ) -> Iterable[Utterance]:
        """ Convert ACS recognized_phrases structure to utterances with correct
            timing. If speaker is provided, returned utterances will match the
            speaker's speech_segs. Otherwise, will have the same grouping as
            given in the recognized_phrases.

            :param recognized_phrases: A list of phrases returned by ACS
            :param speaker: SpeakerInfo. None of no feed-forward track exists
            :returns: An iterable of Utterances
        """
        get_seg = lambda w: (
            int(w['offsetInTicks'] * 1e-4),
            int((w['offsetInTicks'] + w['durationInTicks']) * 1e-4)
        )

        phrase_dicts = [
            dict(
                display=phrase['nBest'][0]['display'],
                segment=get_seg(phrase),
                confidence=phrase['nBest'][0]['confidence'],
                word_segments=list(map(get_seg, phrase['nBest'][0]['words'])),
                word_confidences=[w['confidence'] for w in phrase['nBest'][0]['words']],
                speaker_id=str(phrase.get('speaker', '0'))
            )
            for phrase in recognized_phrases
        ]

        if speaker is None:
            return [Utterance(**d) for d in phrase_dicts]

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

        for phrase_dict in phrase_dicts:
            t0, t1 = phrase_dict['segment']
            # Get index of the segment containing the region start time,
            #  and offset utterance time information by appropriate delta
            i = min(bisect_right(segdivs, t0), len(segdivs) - 1)
            d = offsets[i]
            phrase_dict.update(
                words=phrase_dict['display'].strip().split(),
                segment=(t0+d, t1+d),
                word_segments=[(t0+d, t1+d) for t0, t1 in phrase_dict['word_segments']]
            )

        Word = namedtuple('Word', ['display', 'segment', 'confidence'])
        word_gen = (
            Word(*word_info)
            for p in phrase_dicts
            for word_info in zip(p['words'], p['word_segments'], p['word_confidences'])
        )

        utterances: List[Utterance] = []
        for t0, t1 in speaker.speech_segs:
            # Advance the word generator to overlap with the DIA region. Word
            #  regions must start within the DIA regions.
            word_gen = dropwhile(lambda w: w.segment[0] < t0, word_gen)

            # Get the word regions contained in the DIA region, construct an
            #  Utterance object from them. Make sure not to advance the word
            #  generator and skip words.
            word_gen, seg_words = tee(word_gen, 2)
            seg_words = list(takewhile(lambda w: w.segment[0] < t1, seg_words))
            if not seg_words:
                continue
            display, segs, confs = zip(*seg_words)
            display = ' '.join(display)
            utterances.append(Utterance(
                display=display,
                segment=(t0, t1),
                confidence=sum(confs)/len(confs),
                word_segments=segs,
                word_confidences=confs,
                speaker_id=speaker.speaker_id
            ))

        return utterances

    def process_audio(self, job_config: AzureJobConfig) -> List[mpf.AudioTrack]:
        self.acs.update_acs(job_config.server_info)

        logger.debug(f'Loading file: {job_config.target_file}')
        audio_bytes = mpf_util.transcode_to_wav(
            str(job_config.target_file),
            start_time=job_config.start_time,
            stop_time=job_config.stop_time,
            segments=job_config.speaker.speech_segs if job_config.speaker else None
        )

        try:
            logger.info('Uploading file to blob')
            recording_id = str(uuid.uuid4())
            recording_url = self.acs.upload_file_to_blob(
                audio_bytes=audio_bytes,
                recording_id=recording_id,
                blob_access_time=job_config.blob_access_time
            )
        except mpf.DetectionException:
            raise
        except Exception as e:
            raise mpf.DetectionException(
                'Failed to upload file to blob: {}'.format(e),
                mpf.DetectionError.DETECTION_FAILED
            )

        diarize = job_config.diarize
        output_loc = None
        while output_loc is None:
            try:
                logger.info('Submitting speech-to-text job to ACS')
                output_loc = self.acs.submit_batch_transcription(
                    recording_url=recording_url,
                    job_name=job_config.job_name,
                    diarize=diarize,
                    language=job_config.language,
                    expiry=job_config.expiry
                )
            except Exception as e:
                if 'This locale does not support diarization' in str(e):
                    logger.warning(
                        f'Locale "{job_config.language}" does not support diarization. '
                        'Completing job with diarization disabled.')
                    diarize = False
                else:
                    if job_config.cleanup:
                        logger.info('Marking file blob for deletion')
                        self.acs.delete_blob(recording_id)
                    raise

        try:
            logger.info('Retrieving transcription')
            result = self.acs.poll_for_result(output_loc)

            if result['status'] == 'Failed':
                raise mpf.DetectionException(
                    'Transcription failed: {}'.format(result['statusMessage']),
                    mpf.DetectionError.DETECTION_FAILED
                )

            transcription = self.acs.get_transcription(result)
            logger.info('Speech-to-text processing complete')
        finally:
            if job_config.cleanup:
                logger.info('Marking file blob for deletion')
                self.acs.delete_blob(recording_id)
            logger.info('Deleting transcript')
            self.acs.delete_transcription(output_loc)

        recognized_phrases = transcription['recognizedPhrases']
        utterances = self.convert_word_timing(
            recognized_phrases=recognized_phrases,
            speaker=job_config.speaker)

        logger.info('Completed process audio')
        logger.info('Creating AudioTracks')
        audio_tracks = []
        for utt in utterances:
            # Timing information, desegmented if feed-forward track exists
            utterance_start, utterance_stop = utt.segment

            # Build word confidence and segment strings
            word_confidences = ', '.join(f"{c:f}" for c in utt.word_confidences)
            word_segments = ', '.join(f"{t0:d}-{t1:d}" for t0, t1 in utt.word_segments)

            properties = dict(
                SPEAKER_ID=utt.speaker_id,
                TRANSCRIPT=utt.display,
                WORD_CONFIDENCES=word_confidences,
                ISO_LANGUAGE=BCP47_TO_ISO6393.get(job_config.language, 'UNKNOWN'),
                BCP_LANGUAGE=job_config.language,
                DECODED_LANGUAGE=job_config.language,
                WORD_SEGMENTS=word_segments
            )

            if job_config.speaker is not None:
                # If feed-forward track exists, use provided speaker info
                ldict = job_config.speaker.language_scores
                languages = sorted(
                    ldict,
                    key=lambda lab: ldict.get(lab, -1.0),
                    reverse=True
                )
                language_confs = ', '.join(str(ldict[lab]) for lab in languages)
                language_labels = ', '.join(languages)

                # Add or update properties according to fed-forward info
                properties.update(
                    GENDER=job_config.speaker.gender,
                    GENDER_CONFIDENCE=job_config.speaker.gender_score,
                    SPEAKER_LANGUAGES=language_labels,
                    SPEAKER_LANGUAGE_CONFIDENCES=language_confs
                )

            track = mpf.AudioTrack(
                start_time=utterance_start,
                stop_time=utterance_stop,
                confidence=utt.confidence,
                detection_properties=properties
            )
            audio_tracks.append(track)

        logger.info('Completed processing transcription results')
        return audio_tracks
