#!/usr/bin/env python
# -*- coding: utf-8 -*-

from grad_cam import build_VGG16_and_predict, make_heatmap, show_heatmap

def main():
	img_path = 'image_dir/elephant_1.jpg'###CHOOSE AN IMAGE FILE###
	
	x, model, preds = build_VGG16_and_predict(img_path)
	heatmap = make_heatmap(x, model, preds)
	show_heatmap(img_path, heatmap)
	
	return 0

if __name__ == '__main__':
    main()
