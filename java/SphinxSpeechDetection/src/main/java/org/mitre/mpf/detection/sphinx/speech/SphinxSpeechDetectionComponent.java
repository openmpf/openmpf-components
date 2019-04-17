/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

package org.mitre.mpf.detection.sphinx.speech;

import org.apache.commons.io.FilenameUtils;
import org.mitre.mpf.component.api.detection.MPFAudioJob;
import org.mitre.mpf.component.api.detection.MPFAudioTrack;
import org.mitre.mpf.component.api.detection.MPFComponentDetectionError;
import org.mitre.mpf.component.api.detection.MPFDetectionError;
import org.mitre.mpf.component.api.detection.adapters.MPFAudioAndVideoDetectionComponentAdapter;
import org.mitre.mpf.component.api.detection.adapters.MPFAudioDetectionMediaHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static org.apache.commons.io.FileUtils.deleteQuietly;

public class SphinxSpeechDetectionComponent extends MPFAudioAndVideoDetectionComponentAdapter {
	
    private static final Logger LOG = LoggerFactory.getLogger(SphinxSpeechDetectionComponent.class);
    private final MPFAudioDetectionMediaHandler mediaHandler;
    private final SphinxSpeechDetectionProcessor speechProcessor;

    public SphinxSpeechDetectionComponent() {
        mediaHandler = MPFAudioDetectionMediaHandler.getInstance();
        speechProcessor = new SphinxSpeechDetectionProcessor();
    }

    @Override
    public String getDetectionType() {
        return "SPEECH";
    }

    @Override
    public List<MPFAudioTrack> getDetections(MPFAudioJob job) throws MPFComponentDetectionError {

        LOG.debug("jobName = {}, startTime = {}, stopTime = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
                job.getJobName(), job.getStartTime(), job.getStopTime(), job.getDataUri(),
                job.getJobProperties().size(), job.getMediaProperties().size());

        List<MPFAudioTrack> tracks = new LinkedList<>();

        // TODO: Do away with the need to handle the startTime=0, stopTime=-1 convention (which means to process the entire file).
        // There should not be any special semantics associated with the startTime and stopTime.

        // determine actual start and stop times
        Map<String,String> mediaProperties = job.getMediaProperties();

        int newStartTime = job.getStartTime();
        int newStopTime = job.getStopTime();

        if (job.getStopTime() == -1) {
            Integer duration;
            try {
                if (mediaProperties.get("DURATION") == null) {
                    LOG.error("Could not obtain duration.");
                    throw new MPFComponentDetectionError(MPFDetectionError.MPF_MISSING_PROPERTY, "Could not obtain duration.");
                }
                duration = Integer.valueOf(mediaProperties.get("DURATION"));
            } catch (NumberFormatException ex) {
                LOG.error("Could not obtain duration.", ex);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_PROPERTY_IS_NOT_INT, "Could not obtain duration.");
            }
            LOG.debug("For file {}, duration = {} milliseconds.", job.getDataUri(), duration);

            newStopTime = duration;
        }

        // create a name for temporary audio file to be created from video
        // always create the file in tmp directory in case source is server media file in read-only directory
        String directory = System.getenv("MPF_HOME");
        if (directory == null) {
            directory = "/tmp/";
            LOG.warn("Value for environment MPF_HOME not available - storing interim files in {}", directory);
        } else {
            directory += "/tmp/";
        }
        LOG.debug("directory = {}", directory);

        String basename = FilenameUtils.getBaseName(job.getDataUri());
        File target = new File(directory, String.format("%s_%s_%s.wav", basename, newStartTime, newStopTime));

        LOG.debug("Storing temporary audio for analysis in {}", target);

        try {
            // separate the audio from input media in the proper format and save it in a temporary file
            File source = new File(job.getDataUri());
            try {
                mediaHandler.ripAudio(source, target, newStartTime, newStopTime);

            // TODO: Consider handling these exceptions in different ways.
            } catch (IllegalArgumentException | IOException e) {
                String errorMsg = String.format(
                        "Failed to rip the audio from '%s' (startTime = %s, stopTime = %s) to '%s' due to an Exception.",
                        source, newStartTime, newStopTime, target);
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
            }

            // detect and transcribe any spoken English in the audio

            try {
                speechProcessor.processAudio(target, newStartTime, tracks);
            } catch (IOException e) {
                String errorMsg = "Could not initialize Sphinx StreamSpeechRecognizer.";
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_OTHER_DETECTION_ERROR_TYPE, errorMsg, e);
            } catch (NullPointerException e) {
                // if Sphinx doesn't like an input file it will sometimes generate a NPE and just stop
                // this needs to be caught so that processing isn't halted, but there is
                // nothing specific to be done about it since its cause is internal to Sphinx
                String errorMsg = String.format(
                        "NullPointerException encountered during audio processing of file %s.",
                        target.getAbsoluteFile());
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg, e);
            }
        }
        catch (MPFComponentDetectionError mcde) {
            throw mcde;
        }
        catch(Exception e) {
            String errorMsg = String.format("Unhandled exception processing file %d.", target.getAbsoluteFile());
            LOG.error(errorMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_FAILED, errorMsg, e);
        }
        finally {
            // delete the temporary audio file without throwing any exceptions
            // TODO: may want to keep file around in case it is used again, and clear out files in the temp directory when the system is shut down
            deleteQuietly(target);
        }

        LOG.debug("target = {}, startTime = {}, stopTime = {}, tracks size = {}",
                target, newStartTime, newStopTime, tracks.size());
        
        return tracks;
    }
}
