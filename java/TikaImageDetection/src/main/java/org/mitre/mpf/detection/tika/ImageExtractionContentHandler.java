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


import org.xml.sax.Attributes;
import org.apache.tika.sax.ToTextContentHandler;


public class ImageExtractionContentHandler extends ToTextContentHandler{

    private String pageTag;
    protected int pageNumber;
    private boolean skipTitle;

    public ImageExtractionContentHandler(){
        super();
        pageTag = "div";
        pageNumber = 0;
    }

    public void startElement (String uri, String localName, String qName, Attributes atts) {
        if (pageTag.equals(qName) && (atts.getValue("class").equals("page"))) {
           startPage();
        }
        if (pageTag.equals(qName) && (atts.getValue("class").equals("slide-content"))) {
            startPage();
        }
    }


    public void endElement (String uri, String localName, String qName) {
        if (pageTag.equals(qName)) {
            endPage();
        }
    }

    public void characters(char[] ch, int start, int length) {}


    protected void startPage() {
        pageNumber++;
    }

    protected void endPage() {}

    public String toString(){
        return String.valueOf(pageNumber);
    }

}
