# Localisation Networks bias shapes

import loc_W_shapes as W

locNet1_b = {
    'bc1' : [W.locNet1_W['wc1'][3]],
    'bc2' : [W.locNet1_W['wc2'][3]],
    'bd1' : [W.locNet1_W['wd1'][1]],
	'bd2' : [W.locNet1_W['wd2'][1]],
    'out' : [W.locNet1_W['out'][1]]
}

locNet2_b = {
    'bc1' : [W.locNet2_W['wc1'][3]],
    'bc2' : [W.locNet2_W['wc2'][3]],
    'bd1' : [W.locNet2_W['wd1'][1]],
	'bd2' : [W.locNet2_W['wd2'][1]],
    'out' : [W.locNet2_W['out'][1]]
}

locNet3_b = {
    'bc1' : [W.locNet3_W['wc1'][3]],
    'bc2' : [W.locNet3_W['wc2'][3]],
    'bd1' : [W.locNet3_W['wd1'][1]],
	'bd2' : [W.locNet3_W['wd2'][1]],
    'out' : [W.locNet3_W['out'][1]]
}

locNet4_b = {
    'bc1' : [W.locNet4_W['wc1'][3]],
    'bc2' : [W.locNet4_W['wc2'][3]],
    'bd1' : [W.locNet4_W['wd1'][1]],
	'bd2' : [W.locNet4_W['wd2'][1]],
    'out' : [W.locNet4_W['out'][1]]
}