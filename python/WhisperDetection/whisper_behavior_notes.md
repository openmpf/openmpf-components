Whisper Multi-Lingual Audio Investigation

When set to transcription mode, sometimes Whisper would also translate
the audio

Translating when it's supposed to be transcribing seems to happen more
with larger models and when the audio language is specified, where the
audio gets translated to the specified language. Spanish+English audio
where the model is told the language is in Spanish will have the English
audio translated to Spanish and vice-versa. These translations are very
similar to the translation output when the model is set to actually
translate.

Example: First half of audio is in Spanish, `second half in English`

Default settings (base size multilingual model, model auto-detects
language) output:

-   \'Me comunico con diversas personas en inglés y en español o
    mezclando ambas lenguas. Hay tantas razones. Yo lo hago por
    obviamente solidaridad, porque me reciben en esa comunidad. Como
    crecién el lado mexicano no acostumbraba en ese pasado. Luego, al
    cruzar la frontera metafórica porque no existe derecho, me di
    cuenta, hablan diferente, salpican verdad su conversación con
    palabras en inglés y porque no. Entonces no es fácil hacerlo porque
    he tratado de hacerlo y lo aprecio y lo entiendo mucho más por la
    experiencia que he tenido todos estos años. Y lo hago para tratar de
    pertenecer, para no ser diferente que me consideren, como parte de
    esa comunidad. `Gracias por crear este estilo y ver cómo se trabaja
    Speaking inont不錯. O sea, el flour departador de la creación имеет
    su'`

-   `Highlighted part` is English audio, returns different results each
    run

-   Detected language is Spanish

Large multilingual model, auto-detect language output:

-   \'Me comunico con diversas personas en inglés y en español o
    mezclando ambas lenguas. Hay tantas razones. Yo lo hago por
    obviamente solidaridad, porque me reciben en esa comunidad. Como
    crecí en el lado mexicano, no acostumbraba en ese pasado. Luego, al
    cruzar la frontera metafórica, porque no existe de hecho, me di
    cuenta que hablan diferente. Salpican su conversación con palabras
    en inglés. ¿Y por qué no? Entonces, no es fácil hacerlo, porque he
    tratado de hacerlo. Lo aprecio y lo entiendo mucho más por la
    experiencia que he tenido todos estos años. Y lo hago para tratar de
    pertenecer, para no ser diferente, que me consideren como parte de
    esa comunidad. `Lo hago todo el tiempo. En la escuela, con mis
    colegas. A veces, cuando estoy enseñando, no mezclo tanto. Trato de
    hablar más español en mis clases, porque estoy enseñando en español.
    Si trabajo con gente que son amigas de mí, que hablan español, yo le
    digo la palabra en la lengua que sea más fácil.`

-   Highlighted part is English audio, which has been translated to
    Spanish. Returns the same/almost the same each run

-   Detected language is Spanish

Say we wanted to transcribe the English portion of the audio, so we tell
the model the audio is in English

Base multilingual model, English audio output:

-   `"I communicate with different people in English and Spanish or in
    English. There are so many reasons. I do it because I obviously
    solidarity because I receive in that community. As I grew up in
    Mexico, I didn\'t use that language. Then, when crossing the border,
    I was afraid because there was no right. I realized that speaking
    different people, they really get their conversation with English
    words. I didn\'t do it because I didn\'t do it. I didn\'t try to do
    it because I was afraid. I appreciated it and I understand it much
    more because of the experience I had in all these years. I do it to
    try to keep it from being different, not to be considered as part of
    that community.` I do all the time. At school with my colleagues.
    Usually when I\'m teaching, I don\'t mix as much. I try to speak
    more Spanish in my Spanish classes because we\'re teaching in
    Spanish. I will mix if I\'m working with people that are friends of
    mine, that are bilinguals, say the word in the language that comes
    easiest.\"

-   Highlighted part is Spanish audio, and it's been translated to
    English

Large multilingual model, English audio output:

-   `"I communicate with different people in English and Spanish or
    mixing both languages. There are so many reasons. I do it for
    solidarity, because they receive me in that community. As I grew up
    in the Mexican side, I wasn\'t used to that past. Then, when I
    crossed the metaphorical border, because it doesn\'t exist, I
    realized that they speak differently. They speak in English, and why
    not? So, it\'s not easy to do it, because I\'ve tried to do it. I
    appreciate it and understand it more because of the experience I\'ve
    had all these years. I do it to try to belong, not to be different.`
    I do it all the time, at school with my colleagues. Usually when
    I\'m teaching, I don\'t mix as much. I try to speak more Spanish in
    my Spanish classes, because we\'re teaching in Spanish. I will mix
    if I\'m working with people that are friends of mine, that are
    bilingual, I\'ll say the word in the language that comes easiest.\"

-   Again highlighted is Spanish audio that has been translated to
    English

If we tell the model to translate instead of transcribe, the
translations are very similar to the ones we've seen above

Base multilingual model, auto-detect language translation results:

-   \'I communicate with different people in English and Spanish or
    mixing both languages. There are so many reasons. I do it because
    obviously solidarity, because they receive me in that community. As
    the Mexican people believe, it is not used in that past. Then, when
    crossing the border, metaphorically, because there is no right, I
    realize, talking different, they get out of the truth, their
    conversation, with words in English. Why not? It is not easy to do
    it because I try to do it. I appreciate it and I understand it much
    more because of the experience I had all these years. I do it to try
    to keep it from being, not to be different, and I consider it to be
    a community.\'

-   Detected language is Spanish

-   Only translated Spanish portion of the audio

-   Telling the model the audio is in Spanish returns the same results

Base multilingual model, English audio translation results:

-   `"I communicate with different people in English and Spanish or in
    English. There are so many reasons. I do it because I obviously
    solidarity because I receive in that community. As I grew up in
    Mexico, I didn\'t use that language. Then, when crossing the border,
    I was afraid because there was no right. I realized that speaking
    different people, they really get their conversation with speaking
    English. I didn\'t do it because I didn\'t want to do it. I tried to
    do it because I was afraid. I appreciated it and I understand it
    more because of the experience I had in all these years. I do it to
    try to keep it from being different, not to be considered as part of
    that community.` I do all the time. At school with my colleagues.
    Usually when I\'m teaching, I don\'t mix as much. I try to speak
    more Spanish in my Spanish classes because we\'re teaching in
    Spanish. I will mix if I\'m working with people that are friends of
    mine that are bilingual. Say the word in the language that comes
    easiest.\"

-   Highlighted is Spanish audio

Large multilingual model, auto-detect language translation results:

-   \'I communicate with different people in English and Spanish or
    mixing both languages. There are so many reasons. I do it for
    solidarity, because they receive me in that community. As I grew up
    in the Mexican side, I was not used to that past. Then, when I
    crossed the metaphorical border, because it does not exist, I
    realized that they speak differently. They speak in English, and why
    not? So it is not easy to do it, because I have tried to do it. I
    appreciate it and understand it much more because of the experience
    I have had all these years. I do it to try to belong, not to be
    different, to be considered as part of that community. `I will say
    the word in the language that comes easiest.`\'

-   Highlighted is English audio. It's the last sentence in the English
    audio so the English portion of the audio has been mostly skipped
    over/ignored

Large multilingual model, English audio translation results:

-   `'I communicate with different people in English and Spanish or
    mixing both languages. There are so many reasons. I do it for
    solidarity, because they receive me in that community. As I grew up
    in the Mexican side, I was not used to that past. Then, when I
    crossed the metaphorical border, because it does not exist, I
    realized that they speak differently. They speak in English, and why
    not? So it is not easy to do it, because I have tried to do it. I
    appreciate it and understand it much more because of the experience
    I have had all these years. I do it to try to belong, not to be
    different, to be considered as part of that community.` I think it is
    the language that comes easiest.\'

-   Highlighted is Spanish. Again seems to only want to translate
    Spanish portion of audio but caught the last sentence of the English
    audio as well
