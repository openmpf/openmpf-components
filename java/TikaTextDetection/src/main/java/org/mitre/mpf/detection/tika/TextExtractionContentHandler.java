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

    private static final String PAGE_TAG = "div";
    private static final String CLASS_ATTRIBUTE = "class";
    private static final String SECTION_TAG = "p";
    private static final String PAGE_LABEL = "page";
    private static final String SLIDE_LABEL = "slide-content";
    
    private int _pageNumber;
    private int _sectionNumber;
    private StringBuilder _textResults;
    private ArrayList<ArrayList<StringBuilder>> _pageMap; // TODO: Use HashMap
    private ArrayList<StringBuilder> _currentSectionMap; // TODO: Use HashMap

    // Enable to avoid storing metadata/title text from pdf and ppt documents
    private boolean _skipTitle = true;

    // Disable to skip recording empty sections (warning: could produce an excessive number of empty tracks)
    private final boolean _skipBlankSections = true;

    public TextExtractionContentHandler(){
        super();
        _pageNumber = 0;
        _sectionNumber = 0;
        _textResults = new StringBuilder();
        _pageMap = new ArrayList<>(); // TODO: Use HashMap
        _currentSectionMap = new ArrayList<>(); // TODO: Use HashMap
        _pageMap.add(_currentSectionMap);
        _currentSectionMap.add(new StringBuilder());
    }

    @Override
    public void startElement (String uri, String localName, String qName, Attributes atts) {
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
                resetPage();
            } else {
                startPage();
            }
        }
    }

    @Override
    public void endElement (String uri, String localName, String qName) {}

    @Override
    public void characters(char[] ch, int start, int length) {
        if (length > 0) {
            _textResults.append(ch, start, length);
            _pageMap.get(_pageNumber).get(_sectionNumber).append(ch, start, length);
        }
    }

    private void startPage() {
        _pageNumber++;
        _sectionNumber = 0;
        _currentSectionMap = new ArrayList<>();
        _currentSectionMap.add(new StringBuilder());
        _pageMap.add(_currentSectionMap);
    }

    private void resetPage() {
        _pageNumber = 0;
        _sectionNumber = 0;
        _currentSectionMap.clear();
        _currentSectionMap.add(new StringBuilder());
        _pageMap.clear();
        _pageMap.add(_currentSectionMap);

    }

    private void newSection() {
        if (_skipBlankSections && _currentSectionMap.get(_sectionNumber).toString().trim().isEmpty()){
            return;
        }
        _sectionNumber++;
        _currentSectionMap.add(new StringBuilder());
    }

    @Override
    public String toString(){
        return _textResults.toString();
    }

    // Returns the text detections, subdivided by page number and section.
    public ArrayList<ArrayList<StringBuilder>> getPages(){
        return _pageMap;
    }
}
