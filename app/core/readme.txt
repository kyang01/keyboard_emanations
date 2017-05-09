brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-frei0r --with-libass --with-libvo-aacenc --with-libvorbis --with-libvpx --with-opencore-amr --with-openjpeg --with-opus --with-rtmpdump --with-schroedinger --with-speex --with-theora --with-tools

- make sure no _animation files saved ever
- have animation creation send signals
- add button that can add metdata item to .csv, should be easy
- visualize signal -> popup with trim, framerate, maxframe,... capability 
- create video -> popup with checkbox for each video type
- transform_audio -> popup with option of transofrmation type (cepstrum window, etc) may have none
- threshold_keystrokes-> popup with threshold value, min_dist, max_thresh, etc... preview option that calls visualize signal callback but also includes the clicks in the videos, submit but creates a cluster of keystrokes which is a dropdown under the decoder
- the preview option should compare determined threshold peaks vs actual peaks in blue vs red, with the keystroke text labeled 
- when the keystrokes are determined, have some sort of algorithm link the actual characters (if they exist), probably by comparing peak times and then auto filling in bad keystroke for the mislabeled
- have buttons only appear when on the layer where they are supposed to appear, (cluster keystokes and predict text are only under keystrokes branches), (raw is only one able to transform or create videos)
- cluster_keystrokes -> popup to perform k means clustering on the keystrokes with n classes, etc
- predict_text -> predicts the text after the clusters are created, abstract out transition probability
- hmm should have garbage output, tuned probability that can be adjusted to throw away keystrokes in the hmm