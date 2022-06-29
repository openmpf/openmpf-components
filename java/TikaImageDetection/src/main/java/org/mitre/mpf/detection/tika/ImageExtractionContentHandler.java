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


public class ImageExtractionContentHandler extends DefaultHandler {

    private static final String PAGE_TAG = "div";

    private int _pageNumber;

    // Enable to avoid storing metadata/title text from ppt document.
    private boolean _skipTitle = true;

    @Override
    public void startElement(String uri, String localName, String qName, Attributes atts) {
        if (!PAGE_TAG.equals(qName)) {
            return;
        }

        var classAttr = atts.getValue("class");
        if (classAttr.equals("page")) { // TODO: NPE on jrobble-test.odp
           startPage();
        }
        else if (classAttr.equals("slide-content")) {
            if (_skipTitle) {
                //Skip metadata section of pptx.
                _skipTitle = false;
                //Discard title text. (not part of slide text nor master slide content).
                resetPage();
            } else {
                startPage();
            }
        }
    }

    private void startPage() {
        _pageNumber++;
    }

    private void resetPage() {
        _pageNumber = 0;
    }

    @Override
    public String toString(){
        return String.valueOf(_pageNumber);
    }
}
