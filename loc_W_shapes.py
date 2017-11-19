# Localization Networks Weight Shapes

locNet1_W = {
    'wc1' : [5, 5,   3, 128],
    'wc2' : [5, 5, 128, 192],
    'wd1' : [768, 192],
	'wd2' : [192, 192],
    'out' : [192, 6]
}

locNet2_W = {
	'wc1' : [5, 5,  64, 128],
	'wc2' : [5, 5, 128, 192],
	'wd1' : [192, 192],
	'wd2' : [192, 192],
	'out' : [192, 6]
}
	
locNet3_W = {
	'wc1' : [3, 3, 192, 128],
	'wc2' : [3, 3, 128, 192],
	'wd1' : [3072 , 192],
	'wd2' : [192, 192],
	'out' : [192, 6]
}

locNet4_W = {
	'wc1' : [3, 3, 288, 128],
	'wc2' : [3, 3, 128, 192],
	'wd1' : [3072, 192],
	'wd2' : [192, 192],
	'out' : [192, 6]
}