#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2025 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2025 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

from qwen_speech_summarization_component.qwen_speech_summarization_component import run_component_test

class FakeClass():
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return None 
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__dict__[k] = v

# FakeLLM is a factory that returns an instance where .chat.completions.create is a function with kwargs
# When that function is called, return an array of event-like instances, regardless of arguments
FakeLLM = lambda: FakeClass(chat = FakeClass(completions=FakeClass(create=lambda  *_args, **_kwargs: [ \
            FakeClass(choices=[FakeClass(finish_reason=None, \
                                 delta=FakeClass(content="""{
  "summary": "The conversation is a composite of reflective, instructional, and historical narratives centered on early professional baseball, particularly focusing on the New York Metropolitans and the broader context of Major League Baseball in the 19th and early 20th centuries. It emphasizes player development, physical conditioning, strategic gameplay, and team discipline, drawing on personal anecdotes, historical accounts, and critiques of evolving rules and practices. The discourse includes detailed discussions on batting mechanics, defensive positioning, base running, and the psychological aspects of competition, often referencing legendary players and managers like Jim Mutrie, Amos Rusie, and Dad Clarke. While some segments contain fragmented or unrelated business listings and advertisements, the dominant thread remains a deep engagement with the sport’s history, culture, and professional standards.",
  "primary_topic": "Major League Baseball",
  "other_topics": [
    "Baseball player development",
    "Historical baseball strategies",
    "Personal memoir of a former player",
    "Baseball equipment and field positions",
    "Baseball publishing and promotion",
    "Baseball strategy and player technique",
    "Batting mechanics and stance",
    "Base running tactics",
    "Infield and outfield player selection",
    "Pitcher-batter interaction",
    "Historical evolution of baseball practices",
    "Baseball sliding mechanics",
    "Infield defensive positioning",
    "Advertising and business listings (possibly unrelated)",
    "Baseball defensive strategy and game situations",
    "Baseball fielding and throwing techniques",
    "Baseball gameplay tactics",
    "Pitcher-batter dynamics",
    "Umpiring and rule enforcement",
    "Baseball player health and conditioning",
    "Baseball rules and officiating",
    "Player discipline and routine",
    "Lifestyle habits for athletes",
    "Baseball team management and player welfare",
    "Historical baseball teams and their practices",
    "19th-century New York City commercial establishments",
    "Baseball team performance and player behavior",
    "Hotel and dining establishment",
    "Alcohol consumption and its effects",
    "Travel and scheduling of games",
    "Baseball game performance and atmosphere",
    "Local sports culture in St. Louis",
    "Team morale and player behavior across games",
    "Local business and advertising (beer agent, storage warehouse)",
    "Baseball game narrative involving pitching strategies, player morale, and team dynamics",
    "Moving and packing services in New York City",
    "Historical advertising and business promotion",
    "Historical development of Major League Baseball and the influence of media coverage",
    "Business advertisements for a furniture and carpet store in New York City",
    "Distillery operations and product listings in Baltimore, Maryland",
    "Business advertisement for a liquor and cigar shop in New York City",
    "Historical business and professional listings in New York City",
    "Historical urban signage and commercial advertisements",
    "Local business locations",
    "Transportation and vehicle sales",
    "Beverage and food products",
    "Historical film advertisement",
    "Business and commercial services",
    "Geographic locations in New York City",
    "Film production and distribution",
    "Advertising for a beverage product",
    "Business location and contact information",
    "Product branding and marketing",
    "Historical financial and institutional references",
    "Geographic locations in the United States",
    "Archival or printed material fragments"
  ],
  "classifiers": [
    {
      "classifier": "Major League Baseball",
      "confidence": 0.98,
      "reasoning": "The conversation is consistently centered on Major League Baseball, with extensive discussions of professional players, teams (especially the New York Metropolitans), historical games, strategic techniques, and rule evolution. Specific references to players like Amos Rusie, Dad Clarke, and Jim Mutrie, as well as game scenarios such as the 17-0 victory in St. Louis and the 1884 road trip, confirm the focus on MLB history and practice. The inclusion of detailed gameplay mechanics, player conditioning, and managerial strategies further solidifies this classification. While some segments contain unrelated business listings, the overwhelming majority of content is dedicated to professional baseball, making this the dominant classifier."
    }
  ],
  "entities": {
    "names_of_people": [
      "John J. Troy",
      "Freddie Engel",
      "Jacob Ruppert",
      "T. L. Huston",
      "Harry Stevens",
      "W. A. Sunday",
      "Amos Rusie",
      "Dad Clarke",
      "Hugh Duffy",
      "Johnny Ward",
      "Mike Tiernan",
      "Jerry Denny",
      "Billy Nash",
      "Jimmy Collins",
      "James W.",
      "James Moore",
      "John Lync",
      "J. Fred. Stube Waters",
      "Billy",
      "Mooney",
      "O'Connor",
      "Jim Mutrie",
      "John B. Day",
      "Mr. Lane",
      "M. J. Leonard",
      "Burns Bros.",
      "Big Chief Roseman",
      "Ralph Moore",
      "Tom Bolen",
      "John W.",
      "Diestel",
      "Geo. Ehret",
      "McGinnis",
      "Dave Foutz",
      "Chas H. Nahmmacher",
      "Frank Sparling",
      "Arthur (Kid) Brueks",
      "Billy Newman",
      "Bill Devery",
      "Chris Von der Ahe",
      "Charley Comiskey",
      "Robbins",
      "E. F. Pierce",
      "M. L. Waish",
      "John L. Sullivan",
      "Edward Healey",
      "Jas. Broderick",
      "Bob Reilly",
      "Jac. Philippi",
      "Edward S. McGrath",
      "Sam Fitzpatrick",
      "Robert J. McCartney",
      "MicClement",
      "Sayles",
      "Zahn",
      "Dick Butler",
      "James F. McCaffrey",
      "Howard Crampton",
      "Stewart Paton"
    ],
    "places": [
      "New York City",
      "Polo Grounds",
      "Boston",
      "Detroit",
      "Paterosn, N. J.",
      "2774 Eighth Avenue",
      "1402 Broadway",
      "253 Broadway",
      "103 Park Avenue",
      "220 West 42nd Street",
      "30 East 42nd Street",
      "Harlem",
      "Manhattan Borough",
      "125th Street",
      "129th Street",
      "132d Street",
      "133d Street",
      "154th Street",
      "8928 Morning",
      "Morningside 2727",
      "Old Broadway",
      "156th Street",
      "Harlem River",
      "136 Liberty Street",
      "283 West 132d Street",
      "2490 Eighth Avenue",
      "239 & 241 West 125th St.",
      "5 Main Office and Pockets",
      "Room 209",
      "Rector",
      "William J. Howe Co.",
      "John Wegmann",
      "Haight & Todd",
      "Jersey Real Estate",
      "414-416-418 W. 14th Street",
      "419 West 13th Street S. W. Cor. 53rd St.",
      "8th Ave.",
      "317 West 136th Street",
      "125th Street and Eighth Ave.",
      "226-228 West 125th Street",
      "Morningside 3315",
      "HILL'S =| Colonial SANITARIUM Hotel",
      "ALBERT MUNDORFF, Prop.",
      "THE WEST END",
      "Restaurant and Family Resort",
      "Chelsea 3180",
      "13th Ave. and 30th St. Bet. B’way & 8th Ave.",
      "New York NEW YORK",
      "216 West 46th St.",
      "145th STREET",
      "Lenox Avenue",
      "8th AVE.",
      "142d Street",
      "Audubon",
      "Columbus",
      "St. Louis",
      "61 W. 36th St.",
      "538 W. 38th Street",
      "East 132d St. and Brown PI.",
      "near 133d St. Station St.",
      "50 Church",
      "527 W. 20th St.",
      "Sparling's Storage Warehouse",
      "Flushing",
      "231 Fulton Street",
      "50 Main Street",
      "1634-1636 Broadway",
      "Cor. 50th St.",
      "Winter Garden Building",
      "Broadway Cafe",
      "1235 Second Avenue",
      "2ctc",
      "12 Bridge St.",
      "Sixth Avenue and Tenth Street",
      "622 St. Nicholas Ave",
      "759 SEVENTH AVENUE",
      "50th Street",
      "Seventh Avenue",
      "Sist Street",
      "30th Street",
      "8th Avenue",
      "Morningside Avenue",
      "Cor. 49th St.",
      "2425 8th Ave.",
      "91 Man-",
      "New and Used Cars Sold CORD",
      "5th Avenue",
      "7th Avenue",
      "47th Street",
      "48th Street",
      "52nd Street",
      "6th Avenue",
      "220 West 43nd Street",
      "Toronto",
      "Chicago",
      "Boston",
      "Springfield",
      "Philadelphia",
      "Worcester"
    ],
    "companies": [
      "Troy & Engel",
      "Nicholas Engel Cast-Iron Gas and Water Pipe",
      "Lewis P. Fluhner Company",
      "McDermott & Hanigan",
      "Candler Bungie",
      "Postal Telegraph Building",
      "Daniel Devan & Co.",
      "A. SILZ BASEBALL Incorporated",
      "Poultry & TERP’S’ Game CAFE",
      "Medical D. & J. H. TONJES",
      "Albert Mundorf, Prop.",
      "Hayloft",
      "Gallagher CAFE and Imported and Domestic RESTAURANT",
      "CAFE Mooney & O’Connor’s",
      "J. Fred. Stube Waters CAFE",
      "Daniel Brothers",
      "Clover Valley Print",
      "H. Schwabeland",
      "Tom Butter, Eggs & Cheese",
      "Old English Chop House (Incorporated)",
      "Baseball Magazgime",
      "Moerlein Beer",
      "Sparling's",
      "CARTWRIGHT & CO.",
      "McGrorty’s Furniture and Carpets",
      "Kerin & Dunn",
      "Pure Rye Sandymount Rye Whiskey",
      "Franklin 14 0. V.0",
      "Urbana Wine Co.",
      "Gold Seal",
      "Nuf Said",
      "Sayles, Zahn Company",
      "Construction Company CAFE",
      "Business Men’s Lunch",
      "McDERMOTT & Bohan & DAIRY CO.",
      "O’Beirne",
      "Carstairs Whiskey",
      "B. & N. Bocklish Cigar",
      "James Tappe Bro’s",
      "Cavanaugh",
      "ONE OF THE COALING STATIONS",
      "Telephone 569 Circle",
      "EQUITABLE LIFE ASSURANCE SOCIETY of the UNITED STATES",
      "Chas. A. Stoneham & Company",
      "Universal Film Co.",
      "CA FE",
      "Brokers in All Makes",
      "DRINK KING OF",
      "TABLE WATERS"
    ],
    "body_parts": [
      "arm",
      "hand",
      "foot",
      "shoulder",
      "tendons",
      "brain",
      "left foot",
      "right foot",
      "eyes",
      "wrist",
      "bat",
      "base line",
      "plate",
      "third base",
      "second base",
      "first base",
      "home plate",
      "feet",
      "hip",
      "left arm",
      "throwing arm",
      "bowels",
      "elbow",
      "hips",
      "knee joints",
      "ankles"
    ],
    "organs": [
      "heart",
      "blood"
    ],
    "emotions": [
      "confidence",
      "regret",
      "pride",
      "anxiety",
      "disappointment",
      "excitement",
      "alertness",
      "determination",
      "worry",
      "nostalgia",
      "respect",
      "delight",
      "uneasy",
      "twitching",
      "lack of energy (in the afternoon)",
      "intensity",
      "fighting spirit"
    ]
  }
}"""))], object="chat.completion.chunk"), \
            FakeClass(choices=[FakeClass(finish_reason=True)]), \
        ])))

def test_invocation_with_fake_client():
    result = run_component_test(FakeLLM)
    assert len(result) == 2
    main_detection = result[0]
    classifier_detection = result[1]
    assert main_detection.detection_properties['TEXT'] == "The conversation is a composite of reflective, instructional, and historical narratives centered on early professional baseball, particularly focusing on the New York Metropolitans and the broader context of Major League Baseball in the 19th and early 20th centuries. It emphasizes player development, physical conditioning, strategic gameplay, and team discipline, drawing on personal anecdotes, historical accounts, and critiques of evolving rules and practices. The discourse includes detailed discussions on batting mechanics, defensive positioning, base running, and the psychological aspects of competition, often referencing legendary players and managers like Jim Mutrie, Amos Rusie, and Dad Clarke. While some segments contain fragmented or unrelated business listings and advertisements, the dominant thread remains a deep engagement with the sport’s history, culture, and professional standards."
    assert classifier_detection.detection_properties['CLASSIFIER'] == 'Major League Baseball'
    assert classifier_detection.confidence == 0.98