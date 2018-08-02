package org.mitre.mpf.detection.tika;

import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.Attributes;
import org.apache.tika.sax.ToTextContentHandler;
import java.lang.StringBuilder;
import java.util.ArrayList;

public class MyContentHandler extends ToTextContentHandler {
    private String pageTag = "div";
    protected int pageNumber = 0;
    public StringBuilder text_results;
    public ArrayList<StringBuilder> page_map;
    
    public MyContentHandler(){
        super();
        pageTag = "div";
        pageNumber = 0;
        text_results = new StringBuilder();
        page_map = new ArrayList<StringBuilder>();
        page_map.add(new StringBuilder());
        
    }

    public void startElement (String uri, String localName, String qName, Attributes atts) throws SAXException  {  
        if (pageTag.equals(qName) && (atts.getValue("class").equals("page") || atts.getValue("class").equals("slide-content"))) {
            startPage();
        }
    }


    public void endElement (String uri, String localName, String qName) throws SAXException {  
        if (pageTag.equals(qName)) {
            endPage();
        }
    }
    
    public void characters(char[] ch, int start, int length) throws SAXException {
        if(length>0){
            text_results.append(ch);
            page_map.get(pageNumber).append(ch);
        }
    }
    
    protected void startPage() throws SAXException {
        pageNumber ++;
        page_map.add(new StringBuilder());
    }

    protected void endPage() throws SAXException {
        return;
    }
    
    public String toString(){
        return text_results.toString();
    }
    
    //Returns the text detections, subdivided by page number.
    public ArrayList<StringBuilder> getPages(){
        return page_map;
    }
}
