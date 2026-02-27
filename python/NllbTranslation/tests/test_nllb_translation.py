#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
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

from __future__ import annotations

import json
import logging
import mpf_component_api as mpf
import os
import unittest

from nllb_component import NllbTranslationComponent
from pathlib import Path
from typing import Sequence
from nlp_text_splitter import WtpLanguageSettings

from nllb_component.nllb_translation_component import should_translate
from nllb_component.nllb_utils import NllbLanguageMapper

logging.basicConfig(level=logging.DEBUG)

class TestNllbTranslation(unittest.TestCase):

    #get descriptor.json file path
    cur_path: str = os.path.dirname(__file__)
    descriptorFile: str = os.path.join(cur_path, '../plugin-files/descriptor/descriptor.json')

    #open descriptor.json and save off default props
    with open(descriptorFile) as file:
        descriptor: json = json.load(file)
    descriptorProperties: dict[str, str] = descriptor['algorithm']['providesCollection']['properties']
    defaultProps: dict[str, str] = {}
    for property in descriptor['algorithm']['providesCollection']['properties']:
        defaultProps[property['name']] = property['defaultValue']

    #test translation
    SAMPLE_0 = (
        'Hallo, wie gehts Heute?' # "Hello, how are you today?"
    )
    #expected nllb translation
    OUTPUT_0 = (
        "Hi, how are you today?"
    )
    SAMPLE_1 = (
        'Wie ist das Wetter?' # "How is the weather?"
    )
    OUTPUT_1 = (
        "How's the weather?"
    )
    SAMPLE_2 = (
        'Es regnet.' # "It's raining"
    )
    OUTPUT_2 = (
        "It's raining."
    )

    component = NllbTranslationComponent()

    def test_image_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT= self.SAMPLE_0))
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_image(job)

        props = result[0].detection_properties
        self.assertEqual(self.OUTPUT_0, props["TRANSLATION"])


    def test_paragraph_split_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'por'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        test_generic_job_props['USE_NLLB_TOKEN_LENGTH']='FALSE'
        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'DEFAULT'
        test_generic_job_props['SENTENCE_SPLITTER_NEWLINE_BEHAVIOR'] = 'GUESS'
        test_generic_job_props['SENTENCE_MODEL'] = 'wtp-bert-mini'

        # excerpt from https://www.gutenberg.org/ebooks/16443
        pt_text="""Teimam de facto estes em que são indispensaveis os vividos raios do
nosso desanuviado sol, ou a face desassombrada da lua no firmamento
peninsular, onde não tem, como a de Londres--_a romper a custo um
plumbeo céo_--para verterem alegrias na alma e mandarem aos semblantes o
reflexo d'ellas; imaginam fatalmente perseguidos de _spleen_,
irremediavelmente lugubres e soturnos, como se a cada momento saíssem
das galerias subterraneas de uma mina de _pit-coul_, os nossos alliados
inglezes.

Como se enganam ou como pretendem enganar-nos!

É esta uma illusão ou má fé, contra a qual ha muito reclama debalde a
indelevel e accentuada expressão de beatitude, que transluz no rosto
illuminado dos homens de além da Mancha, os quaes parece caminharem
entre nós, envolvidos em densa atmosphera de perenne contentamento,
satisfeitos do mundo, satisfeitos dos homens e, muito especialmente,
satisfeitos de si.
"""
        ff_track = mpf.GenericTrack(-1, dict(TEXT=pt_text))
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable, to pour joy into the soul and send to the semblances the reflection of them; they imagine fatally pursued from _spleen_,  hopelessly gloomy and dreary, as if every moment they came out of the underground galleries of a pit-coal mine, How they deceive or how they intend to deceive us! is this an illusion or bad faith, against which there is much claim in vain the indelevel and accentuated expression of beatitude, which shines on the illuminated face of the men from beyond the Manch, who seem to walk among us, wrapped in dense atmosphere of perennial contentment, satisfied with the world, satisfied with men and, most of all, satisfied with themselves."
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)
        result_props: dict[str, str] = result_track[0].detection_properties
        print("DEBUG 1")
        print(result_props["TRANSLATION"])
        #self.assertEqual(pt_text_translation, result_props["TRANSLATION"])

        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'SENTENCE'
        test_generic_job_props['SENTENCE_SPLITTER_NEWLINE_BEHAVIOR'] = 'GUESS'
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable to pour joy into the soul and send to the countenances the reflection of them; They imagine themselves fatally haunted by spleen, hopelessly gloomy and sullen, as if at every moment they were emerging from the underground galleries of a pit-coal mine, Our British allies. How they deceive themselves or how they intend to deceive us! Is this an illusion or bad faith, against which there is much to be lamented in vain the indelevel and accentuated expression of beatitude, which shines through the illuminated faces of the men from beyond the Channel, who seem to walk among us, wrapped in a dense atmosphere of perennial contentment, satisfied with the world, satisfied with men and, very especially, satisfied with themselves? Yes , please ."
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)
        #self.assertEqual(pt_text_translation, result_props["TRANSLATION"])
        print("DEBUG 2")
        print(result_props["TRANSLATION"])

        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'DEFAULT'
        test_generic_job_props['SENTENCE_SPLITTER_NEWLINE_BEHAVIOR'] = 'NONE'
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable, to pour joy into the soul and send to the semblances the reflection of them; they imagine fatally pursued from _spleen_,  hopelessly gloomy and sullen, as if at every moment they were emerging from the subterranean galleries of a pit-coal mine, our British allies. How they deceive themselves or how they intend to deceive us! This is an illusion or bad faith, against which much is vainly complained the unlevel and accentuated expression of bliss, which shines through on the face. The European Parliament has been a great help to the people of Europe in the past, and it is a great help to us in the present."
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)
        result_props: dict[str, str] = result_track[0].detection_properties
        #self.assertEqual(pt_text_translation, result_props["TRANSLATION"])
        print("DEBUG 3")
        print(result_props["TRANSLATION"])


    def test_wtp_with_flores_iso_lookup(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'arz'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Arab'
        test_generic_job_props['USE_NLLB_TOKEN_LENGTH']='FALSE'
        test_generic_job_props['SENTENCE_SPLITTER_CHAR_COUNT'] = '100'
        test_generic_job_props['SENTENCE_SPLITTER_INCLUDE_INPUT_LANG'] = 'True'
        test_generic_job_props['FORCE_SENTENCE_SPLITS_FOR_DIFFICULT_LANGUAGES'] = "disabled"

        arz_text="هناك استياء بين بعض أعضاء جمعية ويلز الوطنية من الاقتراح بتغيير مسماهم الوظيفي إلى MWPs (أعضاء في برلمان ويلز). وقد نشأ ذلك بسبب وجود خطط لتغيير اسم الجمعية إلى برلمان ويلز."

        arz_text_translation = "Some members of the National Assembly for Wales were dissatisfied with the proposal to change their functional designation to MWPs (Members of the National Assembly for Wales). This arose from plans to change the name of the assembly to the Parliament of Wales."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=arz_text))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        #self.assertEqual(arz_text_translation, result_props["TRANSLATION"])
        print("DEBUG 4")
        print(result_props["TRANSLATION"])

    def test_long_spanish(self):
        dracula_long_spa ='''
DRÁCULA

Bram Stoker

I. Del diario de Jonathan Harker
Bistritz, 3 de mayo

Salí de Munich a las 8:35 de la noche del primero de mayo, llegando a Viena temprano a la mañana siguiente; debí haber llegado a las 6:46, pero el tren llevaba una hora de retraso. Budapest parece un lugar maravilloso, según el vistazo que pude obtener desde el tren y el poco tiempo que caminé por sus calles. Temí alejarme demasiado de la estación, ya que llegamos tarde y saldríamos lo más cerca posible de la hora fijada.

La impresión que tuve fue que estábamos abandonando el Oeste y entrando en el Este; el más occidental de los espléndidos puentes sobre el Danubio, que aquí es de gran anchura y profundidad, nos condujo a las tradiciones del dominio turco.

Salimos con bastante buen tiempo, y llegamos después del anochecer a Klausenburg. Allí me detuve por la noche en el Hotel Royale. Para la cena, o más bien para la comida nocturna, tomé pollo preparado de algún modo con pimiento rojo, que estaba muy sabroso, pero me dio mucha sed. (Nota: obtener la receta para Mina.) Le pregunté al camarero, y me dijo que se llamaba "paprika hendl," y que, siendo un plato nacional, podría conseguirlo en cualquier lugar de los Cárpatos.

Mis escasos conocimientos de alemán me fueron muy útiles aquí; de hecho, no sé cómo me las habría arreglado sin ellos.

Como tuve algo de tiempo disponible cuando estuve en Londres, visité el Museo Británico e investigué en los libros y mapas de la biblioteca acerca de Transilvania; se me había ocurrido que cierto conocimiento previo del país difícilmente podría dejar de ser importante al tratar con un noble de esa región.

Descubrí que el distrito que él mencionó está en el extremo oriental del país, justo en las fronteras de tres estados: Transilvania, Moldavia y Bukovina, en medio de los montes Cárpatos; una de las partes más salvajes y menos conocidas de Europa.

No pude encontrar ningún mapa ni obra que indicara la localización exacta del castillo de Drácula, ya que no existen mapas en este país que puedan compararse en exactitud con nuestros mapas del Ordnance Survey; sin embargo, descubrí que Bistritz, el pueblo postal mencionado por el conde Drácula, es un lugar bastante conocido. Anotaré aquí algunas de mis notas, ya que podrían refrescar mi memoria cuando relate mis viajes a Mina.

En la población de Transilvania hay cuatro nacionalidades distintas: sajones en el sur, mezclados con los valacos, que son descendientes de los dacios; magiares al oeste y székelys al este y norte. Yo me dirijo hacia estos últimos, quienes afirman ser descendientes de Atila y los hunos. Esto podría ser cierto, ya que cuando los magiares conquistaron el país en el siglo XI encontraron asentados a los hunos.

He leído que todas las supersticiones conocidas del mundo se encuentran reunidas en la herradura de los Cárpatos, como si fuese el centro de una especie de torbellino imaginativo; si es así, mi estancia podría resultar muy interesante. (Nota: Debo preguntarle al conde todo acerca de ellas.)

No dormí bien, aunque mi cama era bastante cómoda, pues tuve toda clase de sueños extraños. Un perro estuvo aullando toda la noche bajo mi ventana, lo que podría haber tenido algo que ver; o quizás fue el paprika, pues tuve que beberme toda el agua de la jarra y aun así seguía sediento. Hacia la mañana logré dormir, y fui despertado por continuos golpes en mi puerta, por lo que supongo que entonces dormía profundamente.

Desayuné más paprika y una especie de gachas de harina de maíz que llamaban "mamaliga," y berenjena rellena de carne picada, un excelente plato que llaman "impletata." (Nota: conseguir también esta receta.)

Tuve que apresurar el desayuno, pues el tren salía poco antes de las ocho, o más bien debería haberlo hecho, ya que después de apresurarme a la estación a las 7:30 tuve que esperar en el vagón durante más de una hora antes de que comenzáramos a movernos.

Me parece que cuanto más al este se viaja, más impuntuales son los trenes. ¿Cómo serán entonces en China?

'''
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'spa'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'


        text_translation = "Some members of the National Assembly for Wales were dissatisfied with the proposal to change their functional designation to MWPs (Members of the National Assembly for Wales). This arose from plans to change the name of the assembly to the Parliament of Wales."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=dracula_long_spa))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        #self.assertEqual(text_translation, result_props["TRANSLATION"])
        print("DEBUG 5")
        print(result_props["TRANSLATION"])

        test_generic_job_props['NLLB_TRANSLATION_TOKEN_SOFT_LIMIT'] = '512'
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        #self.assertEqual(text_translation, result_props["TRANSLATION"])
        print("DEBUG 6")
        print(result_props["TRANSLATION"])




if __name__ == '__main__':
    unittest.main()
