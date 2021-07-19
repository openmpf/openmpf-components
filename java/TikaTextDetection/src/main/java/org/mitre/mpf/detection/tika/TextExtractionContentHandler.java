/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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

import org.apache.tika.sax.ToTextContentHandler;
import org.xml.sax.Attributes;

import java.util.ArrayList;

public class TextExtractionContentHandler extends ToTextContentHandler {
    private static final String pageTag = "div";
    private static final String sectionTag = "p";
    protected int pageNumber;
    protected int sectionNumber;
    public StringBuilder textResults;
    public ArrayList<ArrayList<StringBuilder>> pageMap;
    private ArrayList<StringBuilder> sectionMap;
    private boolean skipTitle;
    private boolean skipBlankSections;

    public TextExtractionContentHandler(){
        super();
        pageNumber = 0;
        sectionNumber = 0;
        // Enable to avoid storing metadata/title text from ppt document.
        skipTitle = true;

        // Disable to skip recording empty sections (warning: could produce an excessive number of empty tracks).
        skipBlankSections = true;

        textResults = new StringBuilder();
        pageMap = new ArrayList<>();
        sectionMap = new ArrayList<>();
        pageMap.add(sectionMap);
        sectionMap.add(new StringBuilder());
    }

    public void startElement (String uri, String localName, String qName, Attributes atts) {
        if (atts.getValue("class") != null) {
            if (pageTag.equals(qName) && (atts.getValue("class").equals("page"))) {
                if (skipTitle) {
                    // Skip metadata section of pdf.
                    skipTitle = false;
                    resetPage();
                } else {
                    startPage();
                }
            }
            if (pageTag.equals(qName) && (atts.getValue("class").equals("slide-content"))) {
                if (skipTitle) {
                    // Skip metadata section of pptx.
                    skipTitle = false;
                    // Discard title text. (not part of slide text nor master slide content).
                    resetPage();
                } else {
                    startPage();
                }
            }
        } else if (sectionTag.equals(qName)) {
            newSection();
        }
    }


    public void endElement (String uri, String localName, String qName) {
        if (pageTag.equals(qName)) {
            endPage();
        }
    }

    public void characters(char[] ch, int start, int length) {
        if (length > 0) {
            textResults.append(ch, start, length);
            pageMap.get(pageNumber).get(sectionNumber).append(ch, start, length);
        }
    }

    protected void startPage() {
        pageNumber ++;
        sectionNumber = 0;
        sectionMap = new ArrayList<>();
        sectionMap.add(new StringBuilder());
        pageMap.add(sectionMap);
    }

    protected void endPage() {}

    protected void resetPage() {
        pageNumber = 0;
        sectionNumber = 0;
        sectionMap.clear();
        sectionMap.add(new StringBuilder());
        pageMap.clear();
        pageMap.add(sectionMap);

    }

    protected void newSection() {
        if (skipBlankSections && sectionMap.get(sectionNumber).toString().trim().isEmpty()){
            return;
        }
        sectionNumber++;
        sectionMap.add(new StringBuilder());
    }

    public String toString(){
        return textResults.toString();
    }

    // Returns the text detections, subdivided by page number and section.
    public ArrayList<ArrayList<StringBuilder>> getPages(){
        return pageMap;
    }
}
