from cleanfid import fid

score = fid.compute_fid(
    fdir1="path/to/real_images",  # 真实图片文件夹
    fdir2="path/to/gen_images",   # 生成图片文件夹
    num_workers=4                 # 并行加载，加速批量处理
)
print(f"FID score: {score}")