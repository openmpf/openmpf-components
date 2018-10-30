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

import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.Attributes;
import org.apache.tika.sax.ToTextContentHandler;
import java.lang.StringBuilder;
import java.util.ArrayList;

public class TextExtractionContentHandler extends ToTextContentHandler {
    private String pageTag = "div";
    protected int pageNumber = 0;
    public StringBuilder textResults;
    public ArrayList<StringBuilder> pageMap;
    private boolean skipTitle;

    public TextExtractionContentHandler(){
        super();
        pageTag = "div";
        pageNumber = 0;
        // Enable to avoid storing metadata/title text from ppt document.
        skipTitle = true;

        textResults = new StringBuilder();
        pageMap = new ArrayList<StringBuilder>();
        pageMap.add(new StringBuilder());

    }

    public void startElement (String uri, String localName, String qName, Attributes atts) throws SAXException  {
        if (pageTag.equals(qName) && (atts.getValue("class").equals("page"))) {
           startPage();
        }
        if (pageTag.equals(qName) && (atts.getValue("class").equals("slide-content"))) {
            if (skipTitle) {
                //Skip metadata section of pptx.
                skipTitle = false;
                //Discard title text. (not part of slide text nor master slide content).
                resetPage();
            } else {
                startPage();
            }
        }
    }


    public void endElement (String uri, String localName, String qName) throws SAXException {
        if (pageTag.equals(qName)) {
            endPage();
        }
    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        if (length > 0) {
            textResults.append(ch);
            pageMap.get(pageNumber).append(ch);
        }
    }

    protected void startPage() throws SAXException {
        pageNumber ++;
        pageMap.add(new StringBuilder());
    }

    protected void endPage() throws SAXException {
        return;
    }

    protected void resetPage() throws SAXException {
        pageNumber = 0;
        pageMap.clear();
        pageMap.add(new StringBuilder());
    }

    public String toString(){
        return textResults.toString();
    }

    // Returns the text detections, subdivided by page number.
    public ArrayList<StringBuilder> getPages(){
        return pageMap;
    }
}
