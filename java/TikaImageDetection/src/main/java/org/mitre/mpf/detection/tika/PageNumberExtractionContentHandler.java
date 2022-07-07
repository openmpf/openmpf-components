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
import org.xml.sax.helpers.DefaultHandler;


public class PageNumberExtractionContentHandler extends DefaultHandler {

    private static final String PAGE_TAG = "div";
    private static final String CLASS_ATTRIBUTE = "class";
    private static final String PAGE_LABEL = "page";
    private static final String SLIDE_LABEL = "slide-content";

    private int _pageNumber;

    @Override
    public void startElement(String uri, String localName, String qName, Attributes atts) {
        if (!PAGE_TAG.equals(qName)) {
            return;
        }
        var classAttr = atts.getValue(CLASS_ATTRIBUTE);
        if (classAttr == null) {
            return;
        }
        if (classAttr.equals(PAGE_LABEL) || classAttr.equals(SLIDE_LABEL)) {
           startPage();
        }
    }

    private void startPage() {
        _pageNumber++;
    }

    @Override
    public String toString(){
        return String.valueOf(_pageNumber);
    }
}
