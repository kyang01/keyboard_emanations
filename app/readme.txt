- threshold_keystrokes->  compare determined threshold peaks vs actual peaks in yellow vs red with the keystroke text labeled, add checkbox of create video
- when the keystrokes are determined, have some sort of algorithm link the actual characters (if they exist), probably by comparing peak times and then auto filling in bad keystroke for the mislabeled
- add text to plot of the character


- cluster_keystrokes -> popup to perform k means clustering on the keystrokes with n classes, etc
- cepstrum pull out before and then slice?
- predict_text -> predicts the text after the clusters are created, abstract out transition probability
- hmm should have garbage output, tuned probability that can be adjusted to throw away keystrokes in the hmm
- start_confuser -> button that starts and listens for you typing on your keyboard and then injects random keystrokes


- fix video mn, mx cutoff
- move create decoder into background
- have buttons only appear when on the layer where they are supposed to appear, (cluster keystokes and predict text are only under keystrokes branches), (raw is only one able to transform or create videos), have hilight the next one to do
-general clip video
- fix determine_offset
- name = folder-file