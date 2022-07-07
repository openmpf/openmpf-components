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

import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class TextExtractionContentHandler extends ToTextContentHandler {

    private static final String PAGE_TAG = "div";
    private static final String CLASS_ATTRIBUTE = "class";
    private static final String SECTION_TAG = "p";
    private static final String PAGE_LABEL = "page";
    private static final String SLIDE_LABEL = "slide-content";
    
    private int _pageNumber;

    private final StringBuilder _allText = new StringBuilder();
    private StringBuilder _sectionText;

    private final Map<Integer, List<StringBuilder>> _pageToSections = new LinkedHashMap<>();

    // Enable to avoid storing metadata/title text from pdf and ppt documents
    private boolean _skipTitle = true;

    public TextExtractionContentHandler(){
        super();
        _pageNumber = 0;
        createPage();
    }

    @Override
    public void startElement(String uri, String localName, String qName, Attributes atts) {
        if (SECTION_TAG.equals(qName)) {
            newSection();
            return;
        }
        var classAttr = atts.getValue(CLASS_ATTRIBUTE);
        if (classAttr == null) {
            return;
        }
        if (PAGE_TAG.equals(qName) &&
                classAttr.equals(PAGE_LABEL) || classAttr.equals(SLIDE_LABEL)) {
            if (_skipTitle) {
                // Skip metadata section of pdf or pptx.
                _skipTitle = false;
                // If slides: Discard title text. (not part of slide text nor master slide content).
                // If pdf: Discard blank page.
                reset();
            } else {
                _pageNumber++;
                createPage();
            }
        }
    }

    @Override
    public void endElement (String uri, String localName, String qName) {}

    @Override
    public void characters(char[] ch, int start, int length) {
        if (length > 0) {
            _allText.append(ch, start, length);
            _sectionText.append(ch, start, length);
        }
    }

    private void createPage() {
        _sectionText = new StringBuilder();
        _pageToSections.put(_pageNumber, new LinkedList<>());
        _pageToSections.get(_pageNumber).add(_sectionText);
    }

    private void reset() {
        _pageNumber = 0;
        _pageToSections.clear();
        createPage();
    }

    private void newSection() {
        // Skip blank sections.
        if (_sectionText.toString().isBlank()) {
            return;
        }
        _sectionText = new StringBuilder();
        _pageToSections.get(_pageNumber).add(_sectionText);
    }

    @Override
    public String toString(){
        return _allText.toString();
    }

    // Returns the text detections, subdivided by page number and section.
    public Map<Integer, List<StringBuilder>> getPages(){
        return _pageToSections;
    }
}
