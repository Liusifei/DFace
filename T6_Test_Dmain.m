%Test_Dmain.m
clear;clc;close all;
%% configuration
fprintf('configuration...\n');
F1_config;

%% read test;
%
fin = fopen(test_po_name,'r');
uv = fscanf(fin, '%f %f',[2,inf]);
fclose(fin);
if max(size(uv))~=83
F2_facelandmarknotate;
error('please restart this app.')
end
uv = uv';
I_test = im2double(imread(test_im_name));
imwrite(I_test,fullfile(resultfolder,'orig.png'),'png');
[r,c,~] = size(I_test);

%% read exemplar
fin = fopen(expr_po_name,'r');
xy = fscanf(fin, '%f %f',[2,inf]);
fclose(fin);
if max(size(xy))~=83
F2_facelandmarknotate;
error('please restart this app.')
end
xy = xy';
I_expr = im2double(imread(expr_im_name));
imwrite(I_expr,fullfile(resultfolder,'expr.png'),'png');
%% warp exemplar to the test

contour = [1:15,80:-1:77,83:-1:81];

lambda = 100;
sz=[r,c];
tps = tps_init(sz, uv, xy, lambda);
warp_exemplar = tps_warpimg(I_expr, tps);

warp_mask = roipoly(zeros(r,c), uv(contour,1), uv(contour,2));
%% get masks
ep_mask = T2_GenerateMasks(I_expr, xy);
mask = T2_GenerateMasks(I_test,uv);
clc;
%%  decompose face layers

Lab_expr = RGB2Lab(warp_exemplar);
Lab_test = RGB2Lab(I_test);

mmask = mask.c1 & warp_mask; % skin
mmmask = repmat(mmask,[1,1,3]);

mean_test = sum(sum(Lab_test .* double(mmmask)))./sum(sum(double(mmmask)));
mean_expr = sum(sum(Lab_expr.* double(mmmask)))./sum(sum(double(mmmask)));
% 
Lab_expr = Lab_expr - repmat((mean_expr - mean_test),[size(Lab_expr,1),size(Lab_expr,2),1]);

%% facial structure transfer

h = fspecial('gaussian',21,10);
nwarp_mask = imfilter(double(1-warp_mask),h);
warpmask = floor(1-nwarp_mask).*warp_mask;
cmask = 1-mask.c1;
test = T1_StructureTransfer(Lab_test, Lab_expr, warpmask);
test = T1_StructureTransfer(test, imfilter(Lab_test,fspecial('gaussian',3,1.6)), mask.nosebo);
warpmask = imfilter(warpmask,h);
test(:,:,2:3) = repmat(warpmask,[1,1,2]) .* Lab_expr(:,:,2:3)+...
    (1-repmat(warpmask,[1,1,2])).* test(:,:,2:3);
test = EdgeSmoothing2(Lab_test, test, cmask,3);

%% lip transfer
trans_lip = Lab_expr(:,:,1);
masklip = imfilter(mask.lip,fspecial('gaussian',3,1.6));
trans(:,:,1) = masklip .* trans_lip + (1-masklip).* test(:,:,1);
trans(:,:,2:3) = repmat(masklip,[1,1,2]) .* Lab_expr(:,:,2:3) +...
    (1-repmat(masklip,[1,1,2])).* test(:,:,2:3);
test = EdgeSmoothing(trans, test, mask.lip + mask.lipcv, 1);

trans_com = im2double(Lab2RGB(test));

figure;imshow(I_test)
figure;imshow(I_expr)
figure;imshow(0.4*I_test + 0.6*trans_com);