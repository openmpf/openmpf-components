/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

package org.mitre.mpf.detection.tika;

import org.mitre.mpf.component.api.detection.*;

import java.util.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;

import java.lang.StringBuilder;

import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.sax.BodyContentHandler;
import org.apache.tika.language.LanguageIdentifier;

import org.apache.tika.language.detect.LanguageDetector;
import org.apache.tika.langdetect.OptimaizeLangDetector;
import org.apache.tika.language.detect.LanguageResult;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.StringWriter;
import java.io.PrintWriter;


public class TikaTextDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger LOG = LoggerFactory.getLogger(TikaTextDetectionComponent.class);
    private static final Map<String, String> lang_map;
    static{
         lang_map = Collections.unmodifiableMap(initLangMap());
    }
    
    // Handles the case where the media is a generic type
    public List<MPFGenericTrack>  getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        LOG.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
            mpfGenericJob.getJobName(), mpfGenericJob.getDataUri(),
            mpfGenericJob.getJobProperties().size(), mpfGenericJob.getMediaProperties().size());
            
        // =========================
        // Tika Detection
        // =========================

        //Specify filename for tika parsers here.
        File file = new File(mpfGenericJob.getDataUri());
        String text_output = "";
        String metadata_output = "";
        String context_output = "";
        List<StringBuilder> page_output = new ArrayList<StringBuilder>();
        
        
        try{
            //Init parser with custom content handler for parsing text per page (PDF/PPTX)
            Parser parser = new AutoDetectParser();
            TextExtractionContentHandler handler = new TextExtractionContentHandler();
            Metadata metadata = new Metadata();
            FileInputStream inputstream = new FileInputStream(file);
            ParseContext context = new ParseContext();

            //Parse file.
            //If the format is .pdf or .pptx, output will be divided by page/slide.
            parser.parse(inputstream, handler, metadata, context);
            metadata_output = metadata.toString();
            context_output = context.toString();
            text_output =  handler.toString();
            page_output = handler.getPages();

        }catch(Exception e){
            LOG.error(extractStackTrace(e));
            LOG.error(e.toString());
            LOG.error("Error parsing file."+"\nFilepath = " + file.toString());
        }
        
        float confidence = -1.0f;
        List<MPFGenericTrack> tracks = new LinkedList<MPFGenericTrack>();
        Map<String,String> properties = mpfGenericJob.getJobProperties();
        
        //Acquire tag file.
        String tag_file = "";
        if(properties.get("TAGGING_FILE") == null) {
            String rundirectory = this.getRunDirectory();
            if(rundirectory == null || rundirectory.length() < 1)
                rundirectory = "../plugins";
            tag_file = rundirectory+"/TikaDetection/config/text-tags.json";
        } else {
            tag_file = properties.get("TAGGING_FILE") ;
        }
        tag_file = this.expandPathEnvVars(tag_file);
        

        //If output exists, separate all output into separate pages.
        //Tag each page by detected language.
        if(page_output.size() >= 1){
            //Language identifier
            OptimaizeLangDetector identifier = new OptimaizeLangDetector();
            JSONObject string_tags = new JSONObject();
            JSONObject regex_tags = new JSONObject();
            try{
                identifier.loadModels();       
            }catch (IOException e) {
                LOG.error(extractStackTrace(e));
                LOG.error(e.toString());
                LOG.error("Failed to load language models.");
            }
            
            //Parse Tag File
            try{
                JSONParser parser = new JSONParser();
                JSONObject a = (JSONObject) parser.parse(new FileReader(tag_file));               
                string_tags = (JSONObject)a.get("TAGS_STRING_SPLIT");
                regex_tags = (JSONObject)a.get("TAGS_REGEX");
            }catch (FileNotFoundException e) {
                LOG.error(extractStackTrace(e));
                LOG.error(e.toString());
                LOG.error("Tag file "+tag_file+" not found!");
            }catch (ParseException e) {
                LOG.error(extractStackTrace(e));
                LOG.error(e.toString());
                LOG.error("Error parsing tag file.");
            }catch (IOException e) {
                LOG.error(extractStackTrace(e));
                LOG.error(e.toString());
                LOG.error("I/O Error!");
            }

            int page_id_len = (int)(java.lang.Math.log10(page_output.size())) + 1;
            for(int i = 0; i < page_output.size(); i++){
                
                Map<String, String> genericDetectionProperties = new HashMap<String, String>();
                Map<String, String> imageDetectionProperties = new HashMap<String, String>();
                List<String> splitTags = new ArrayList<String>();
                List<String> regexTags = new ArrayList<String>();
                //Split out non-alphanumeric characters.
                String pageText = page_output.get(i).toString().toLowerCase();
                String[] values = pageText.split("\\W+");
                
                Set<String> hashSet = new HashSet<String>(Arrays.asList(values));

                try{
                    
                    for(Iterator iterator = string_tags.keySet().iterator(); iterator.hasNext();) {
                        String key = (String) iterator.next();
                        for(Object x: (JSONArray)string_tags.get(key)){
                            if(hashSet.contains(x)){
                                splitTags.add(key);
                                break;
                            }
                        }
                    }
                    
                    for(Iterator iterator = regex_tags.keySet().iterator(); iterator.hasNext();) {
                        String key = (String) iterator.next();
                        for(Object x: (JSONArray)regex_tags.get(key)){
                            if(pageText.matches((String)x)){
                                regexTags.add(key);
                                break;
                            }
                        }
                    }
                    
                    String splitTagsResult = String.join(",", splitTags);
                    String regexTagsResult = String.join(",", regexTags);
                    String text_detect = page_output.get(i).toString();
                    LanguageResult lang_result = identifier.detect(text_detect);
                    String language = lang_result.getLanguage();
                    
                    if(lang_map.containsKey(language))
                        language = lang_map.get(language);
                    if(!lang_result.isReasonablyCertain())
                        language = null;
                    genericDetectionProperties.put("TEXT", text_detect);
                    
                    if(language != null && language.length()>0)
                        genericDetectionProperties.put("TEXT_LANGUAGE",  language);
                    if(splitTagsResult.length() > 0)
                        genericDetectionProperties.put("TAGS_STRING", splitTagsResult);
                    if(regexTagsResult.length() > 0)
                        genericDetectionProperties.put("TAGS_REGEX", regexTagsResult);
                    
                } catch (Exception e){
                    LOG.error(extractStackTrace(e));
                    LOG.error(e.toString());
                    LOG.error("Failed to process text detections.");
                }
                genericDetectionProperties.put("PAGE_NUM",String.format("%0"+String.valueOf(page_id_len)+"d", i));
                MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                tracks.add(genericTrack);
            }
        }
        LOG.debug(
                String.format("[%s] Processing complete. Generated %d generic tracks.",
                        mpfGenericJob.getJobName(),
                        tracks.size()));

        return tracks;
    }
    
    //Map for translating from ISO 639-2 code to english description.
    private static Map<String,String> initLangMap(){
        Map<String,String> map = new HashMap<String,String>();
        map.put("af","Afrikaans");
        map.put("an","Aragonese");
        map.put("ar","Arabic");
        map.put("ast","Asturian");
        map.put("be","Belarusian");
        map.put("br","Breton");
        map.put("bg","Bulgarian");
        map.put("bn","Bengali");
        map.put("ca","Catalan");
        map.put("cs","Czech");
        map.put("cy","Welsh");
        map.put("da","Danish");
        map.put("de","German");
        map.put("el","Greek");
        map.put("en","English");
        map.put("eo","Esperanto");
        map.put("es","Spanish");
        map.put("et","Estonian");
        map.put("eu","Basque");
        map.put("fa","Persian");
        map.put("fi","Finnish");
        map.put("fr","French");
        map.put("ga","Irish");
        map.put("gl","Galician");
        map.put("gu","Gujarati");
        map.put("he","Hebrew");
        map.put("hi","Hindi");
        map.put("hr","Croatian");
        map.put("ht","Haitian");
        map.put("hu","Hungarian");
        map.put("id","Indonesian");
        map.put("is","Icelandic");
        map.put("it","Italian");
        map.put("ja","Japanese");
        map.put("km","Khmer");
        map.put("kn","Kannada");
        map.put("ko","Korean");
        map.put("lt","Lithuanian");
        map.put("lv","Latvian");
        map.put("mk","Macedonian");
        map.put("ml","Malayalam");
        map.put("mr","Marathi");
        map.put("ms","Malay");
        map.put("mt","Maltese");
        map.put("ne","Nepali");
        map.put("nl","Dutch");
        map.put("no","Norwegian");
        map.put("oc","Occitan");
        map.put("pa","Punjabi");
        map.put("pl","Polish");
        map.put("pt","Portuguese");
        map.put("ro","Romanian");
        map.put("ru","Russian");
        map.put("sk","Slovak");
        map.put("sl","Slovenian");
        map.put("so","Somali");
        map.put("sq","Albanian");
        map.put("sr","Serbian");
        map.put("sv","Swedish");
        map.put("sw","Swahili");
        map.put("ta","Tamil");
        map.put("te","Telugu");
        map.put("th","Thai");
        map.put("tl","Tagalog");
        map.put("tr","Turkish"); 
        map.put("uk","Ukrainian");
        map.put("ur","Urdu");
        map.put("vi","Vietnamese");
        map.put("wa","Walloon");
        map.put("yi","Yiddish");
        map.put("zh-cn","Simplified Chinese");
        map.put("zh-tw","Traditional Chinese");
        return map;
    }
    


    //TODO: Delete once integrated into OpenMPF-Java-Components
    //Utility function
    //Expands environment variables in a given path/filename.
    public static String expandPathEnvVars(String path) {
        Map<String, String> sysEnv = System.getenv();
        for (Map.Entry<String, String> entry : sysEnv.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            path = path.replaceAll("\\$" + key, value);
        }
        return path;
    }

    //Extract stack trace as a string
    public static String extractStackTrace(Exception e){
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        e.printStackTrace(pw);
        return sw.toString();
    }
    
    // The TikeDetection component supports generic file types (pdfs, documents, txt, etc.)
    public boolean supports(MPFDataType mpfDataType) {
        return mpfDataType != null && MPFDataType.UNKNOWN.equals(mpfDataType);
    }

    public String getDetectionType() {
        return "TEXT";
    }
    
    public List<MPFImageLocation> getDetections(MPFImageJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE, "Image detection not supported.");
    }
    
    public List<MPFVideoTrack> getDetections(MPFVideoJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE, "Video detection not supported.");
    }
    
    public List<MPFAudioTrack> getDetections(MPFAudioJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE, "Audio detection not supported.");
    }
}
