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

    def segments_str(self, start_offset: int):
        return ', '.join(
            f"{t0 + start_offset:d}-{t1 + start_offset:d}"
            for t0, t1 in self.word_segments
        )

    def confidences_str(self):
        return ', '.join(str(c) for c in self.word_confidences)


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

        phrase_dicts = []
        for phrase in recognized_phrases:
            display = phrase['nBest'][0]['display'].strip()
            segment = get_seg(phrase)
            confidence = phrase['nBest'][0]['confidence']
            word_segments = list(map(get_seg, phrase['nBest'][0]['words']))
            word_confidences = [w['confidence'] for w in phrase['nBest'][0]['words']]
            speaker_id = str(phrase.get('speaker', '0'))

            # Ensure display text tokens are one-to-one with word segments
            #  If not, replace with bare words. This loses punctuation and
            #  capitalization, but keeps correspondence between detection props
            if len(display.split()) != len(word_segments):
                logger.warning(
                    "Display text did not match number of segments, using bare "
                    "words instead (will lose punctuation and capitalization "
                    "information)")
                display = ' '.join(w['word'].strip() for w in phrase['nBest'][0]['words'])

            phrase_dicts.append(dict(
                display=display,
                segment=segment,
                confidence=confidence,
                word_segments=word_segments,
                word_confidences=word_confidences,
                speaker_id=speaker_id
            ))

        # If there are no feed-forward speech segments, then each phrase
        #  is its own utterance
        if speaker is None:
            return [Utterance(**d) for d in phrase_dicts]

        # Preserve only word-level information, other utterance data is
        #  determined by feed-forward speaker information
        words = [w for p in phrase_dicts for w in p['display'].split()]
        word_segments = [s for p in phrase_dicts for s in p['word_segments']]
        word_confidences = [c for p in phrase_dicts for c in p['word_confidences']]

        # Create parallel lists for the beginning and length of each
        #  feedforward segment, so that word timestamps can be shifted
        #  to be relative to the beginning of the input audio file.
        segbegs, seglens = zip(*(
            (t0, t1-t0) for t0, t1 in sorted(speaker.speech_segs)
        ))

        # The dividing line between each segment relative to regions
        segdivs = list(accumulate(seglens))

        # The offset to add to region times, according to their segment
        offsets = [a-b for a, b in zip(segbegs, [0] + segdivs)]

        # Offset time for each word
        word_deltas = [
            offsets[min(bisect_right(segdivs, t0), len(segdivs) - 1)]
            for t0, t1 in word_segments
        ]
        word_segments = [(t0+d, t1+d) for (t0, t1), d in zip(word_segments, word_deltas)]

        # Collect word information into named tuple for easy access
        Word = namedtuple('Word', ['display', 'segment', 'confidence'])
        words = [Word(*w) for w in zip(words, word_segments, word_confidences)]
        words = sorted(words, key=lambda w: w.segment)
        word_gen = (w for w in words)

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

        missing_models = set()
        default_locale = job_config.language
        if (lang := job_config.override_default_language) is not None:
            if lang in ISO6393_TO_BCP47:
                for locale in ISO6393_TO_BCP47[lang]:
                    if locale in self.acs.supported_locales:
                        logger.debug(
                            f"Override default language ('{lang}') detected, "
                            f"using its corresponding BCP-47 locale "
                            f"('{locale}') in place of component default "
                            f"('{default_locale}')."
                        )
                        default_locale = locale
                        break
                else:
                    logger.warning(
                        f"Override default language '{lang}' was provided, but "
                        f"no corresponding BCP-47 language codes are supported "
                        f"by Azure Speech-to-Text. Using component default "
                        f"('{default_locale}') instead."
                    )
                    missing_models.add(lang)
            else:
                logger.warning(
                    f"Override default language '{lang}' was provided, but does "
                    f"not correspond to a BCP-47 language code. Using component "
                    f"default ('{default_locale}') instead."
                )
                missing_models.add(lang)

        locale = default_locale
        if job_config.speaker is not None:
            speaker_language_valid = False
            if (lang := job_config.speaker.language) in ISO6393_TO_BCP47:
                for locale in ISO6393_TO_BCP47[lang]:
                    if locale in self.acs.supported_locales:
                        speaker_language_valid = True
                        break

            if not speaker_language_valid:
                missing_models.add(job_config.speaker.language)
                ldict = job_config.speaker.language_scores
                for lang in sorted(ldict.keys(), key=ldict.get, reverse=True):
                    if lang in ISO6393_TO_BCP47:
                        for locale in ISO6393_TO_BCP47[lang]:
                            if locale in self.acs.supported_locales:
                                logger.warning(
                                    f"Language supplied in feed-forward track "
                                    f"('{job_config.speaker.language}') is not "
                                    f"supported. Transcribing with highest-scoring "
                                    f"language ('{lang}', using '{locale}' "
                                    f"locale) instead."
                                )
                                break
                        else:
                            # If for loop completes with no break, lang had no
                            #  supported corresponding locale
                            continue
                        # If we get here, a supported locale was found; break
                        break

                    # Language is either not a valid ISO 639-3 code, does not
                    #  have a corresponding BCP-47 code, or is not supported
                    #  by Azure Speech. Add to MISSING_LANGUAGE_MODELS
                    missing_models.add(lang)
                else:
                    logger.warning(
                        f"Neither the language supplied in feed-forward "
                        f"track ('{job_config.speaker.language}'), nor any "
                        f"other detected language are supported. "
                        f"Transcribing with default language "
                        f"('{default_locale}') instead."
                    )
                    locale = default_locale

        if locale not in self.acs.supported_locales:
            raise mpf.DetectionException(
                f"Selected locale ('{locale}') is not supported by Azure "
                f"Speech-to-Text.",
                mpf.DetectionError.DETECTION_FAILED
            )

        diarize = job_config.diarize
        output_loc = None
        while output_loc is None:
            try:
                logger.info(f'Submitting {locale} speech-to-text job to ACS')
                output_loc = self.acs.submit_batch_transcription(
                    recording_url=recording_url,
                    job_name=job_config.job_name,
                    diarize=diarize,
                    language=locale,
                    expiry=job_config.expiry
                )
            except Exception as e:
                if 'This locale does not support diarization' in str(e):
                    logger.warning(
                        f'Locale "{locale}" does not support diarization. '
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
                error_info = result.get('properties', {}).get('error', {})
                code = error_info.get('code')
                msg = error_info.get('message')
                raise mpf.DetectionError.DETECTION_FAILED.exception(
                        f'Transcripton failed with code "{code}" and message "{msg}".')

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

            properties = dict(
                SPEAKER_ID=utt.speaker_id,
                TRANSCRIPT=utt.display,
                WORD_CONFIDENCES=utt.confidences_str(),
                ISO_LANGUAGE=BCP47_TO_ISO6393.get(locale, 'UNKNOWN'),
                BCP_LANGUAGE=locale,
                DECODED_LANGUAGE=locale,
                WORD_SEGMENTS=utt.segments_str(job_config.start_time),
                MISSING_LANGUAGE_MODELS=', '.join(missing_models)
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
                start_time=utterance_start + job_config.start_time,
                stop_time=utterance_stop + job_config.start_time,
                confidence=utt.confidence,
                detection_properties=properties
            )
            audio_tracks.append(track)

        logger.info('Completed processing transcription results')
        return audio_tracks
