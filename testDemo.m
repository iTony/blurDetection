clear;
clc;

% img = imread('lena0.png');
% img = imread('lena.png');
% img = imread('chicky_512.png');
% img = imread('baboon.jpg');
% img = imread('HappyFish.jpg');
img = imread('10.jpg');
% img = imread('11.jpg');
% img = imread('12.jpg');

if ndims(img) > 2
    img = rgb2gray(img);
end

% % gaussian blur
% sigma = 0.01;
% gfilter = fspecial('gaussian', [5 5], sigma);
% img = imfilter(img, gfilter, 'replicate');
% imshow(img);

[unblured, BlurExtent] = blurDetection(img, 35, 0.05);

unblured
BlurExtent
