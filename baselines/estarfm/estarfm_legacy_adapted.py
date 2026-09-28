from utils import read_raster, writeimage
import math
import numpy as np
try:
    from osgeo import gdal
except ImportError:
    import gdal
import os
import datetime
import idlwrap
from scipy.special import fdtrc


def analytic_ols_slope_with_pvalue(x, y):
    """一元OLS解析解，避免逐像元调用statsmodels/SVD导致Windows崩溃。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = x.size
    if n < 3:
        return np.nan, 1.0

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    ss_x = float(np.dot(x_centered, x_centered))
    ss_y = float(np.dot(y_centered, y_centered))
    if ss_x <= 1e-15 or ss_y <= 1e-15:
        return np.nan, 1.0

    cross = float(np.dot(x_centered, y_centered))
    slope = cross / ss_x
    r2 = min(max((cross * cross) / (ss_x * ss_y), 0.0), 1.0)
    if r2 >= 1.0 - 1e-15:
        p_value = 0.0
    else:
        f_statistic = (r2 / (1.0 - r2)) * (n - 2)
        p_value = float(fdtrc(1, n - 2, f_statistic))
    return slope, p_value

# 🚨 为单波段数据特制的“强制升维”包裹函数
# 🚨 修复版：强制统一背景值，解决边缘切块崩溃问题
def safe_read_raster(path):
    nl, ns, data = read_raster(path)

    # 1. 把所有 GEE 产生的 NaN (空值) 强制转换为 ESTARFM 的标准背景值
    data = np.nan_to_num(data, nan=-9999.0)

    # 合法FPAR=0必须保留。输入准备阶段已经把NoData显式统一为-9999。

    # 3. 升维兼容
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    return nl, ns, data

def run_estarfm_core(path_F1, path_C1, path_F2, path_C2, path_C0, temp_dir):
    """
    单波段兼容版 ESTARFM 核心函数
    """
    print(f"   [ESTARFM] 引擎已启动，正在执行空间切块和相似像元搜索 (请耐心等待)...")
    
    w = 25              # 半窗口大小，实际窗口为 51x51
    num_class = 5       # 土地覆盖分类数
    DN_min = 0.0        
    DN_max = 1.0        
    background = -9999  
    patch_long = 500    
    
    path1 = path_F1
    path2 = path_C1
    path3 = path_F2
    path4 = path_C2
    path5 = path_C0
    temp_file = temp_dir

    if not os.path.exists(temp_file):
        os.makedirs(temp_file)

    suffix = os.path.splitext(path1)[-1]
    # 替换为 safe_read_raster
    nl, ns, FileName1 = safe_read_raster(path1)
    orig_ns, orig_nl = ns, nl
    fp_ds = gdal.Open(path1)
    nb = fp_ds.RasterCount

    n_nl = math.ceil(orig_nl / patch_long)
    n_ns = math.ceil(orig_ns / patch_long)
    ind_patch = np.zeros((n_nl * n_ns, 4), dtype=int)

    for i_ns in range(0, n_ns):
        for i_nl in range(0, n_nl):
            ind_patch[n_ns * i_nl + i_ns, 0] = i_ns * patch_long
            ind_patch[n_ns * i_nl + i_ns, 1] = np.min([ns - 1, (i_ns + 1) * patch_long - 1])
            ind_patch[n_ns * i_nl + i_ns, 2] = i_nl * patch_long
            ind_patch[n_ns * i_nl + i_ns, 3] = np.min([nl - 1, (i_nl + 1) * patch_long - 1])

    def save_patches(path_img, prefix):
        # 替换为 safe_read_raster
        _, _, img_data = safe_read_raster(path_img)
        tempoutname = os.path.join(temp_file, prefix)
        for isub in range(n_nl * n_ns):
            c1, c2, r1, r2 = ind_patch[isub, 0], ind_patch[isub, 1], ind_patch[isub, 2], ind_patch[isub, 3]
            data = img_data[:, r1:r2 + 1, c1:c2 + 1]
            out_name = tempoutname + str(isub + 1) + suffix
            writeimage(data, out_name, path_img)

    save_patches(path1, 'temp_F1')
    save_patches(path2, 'temp_C1')
    save_patches(path3, 'temp_F2')
    save_patches(path4, 'temp_C2')
    save_patches(path5, 'temp_C0')

    for isub in range(0, n_nl * n_ns):
        # 替换所有的读取为 safe_read_raster
        nl_p, ns_p, fine1 = safe_read_raster(os.path.join(temp_file, f'temp_F1{isub + 1}{suffix}'))
        _, _, coarse1 = safe_read_raster(os.path.join(temp_file, f'temp_C1{isub + 1}{suffix}'))
        _, _, fine2 = safe_read_raster(os.path.join(temp_file, f'temp_F2{isub + 1}{suffix}'))
        _, _, coarse2 = safe_read_raster(os.path.join(temp_file, f'temp_C2{isub + 1}{suffix}'))
        _, _, coarse0 = safe_read_raster(os.path.join(temp_file, f'temp_C0{isub + 1}{suffix}'))

        fine0 = np.full((nb, nl_p, ns_p), background, dtype=float)
        row_index = np.tile(np.arange(nl_p)[:, None], (1, ns_p))
        col_index = np.tile(np.arange(ns_p), (nl_p, 1))

        uncertain = (DN_max * 0.002) * np.sqrt(2)
        similar_th = np.zeros((2, nb)).astype(float)
        for iband in range(0, nb):
            valid_f1 = fine1[iband][fine1[iband] != background]
            valid_f2 = fine2[iband][fine2[iband] != background]
            similar_th[0, iband] = np.std(valid_f1) * 2.0 / num_class
            similar_th[1, iband] = np.std(valid_f2) * 2.0 / num_class

        D_temp1 = w - np.tile((idlwrap.indgen(w*2+1)), (int(w*2+1), 1))
        d1 = np.power(D_temp1, 2)
        D_temp2 = w - np.tile(idlwrap.indgen(1, w*2+1), (1, int(w*2+1)))
        d2 = np.power(D_temp2, 2)
        D_D_all = (1.0 + np.sqrt(d1 + d2) / float(w)).flatten()

        valid_index = np.zeros((nl_p, ns_p)).astype(int)
        ind_valid = np.where((fine1[0] != background) & (fine2[0] != background) & 
                             (coarse1[0] != background) & (coarse2[0] != background) & 
                             (coarse0[0] != background))
        if len(ind_valid[0]) > 0:
            valid_index[ind_valid] = 1

        for j in range(0, nl_p):
            for i in range(0, ns_p):
                if valid_index[j, i] == 1:
                    ai = int(np.max([0, i - w]))
                    bi = int(np.min([ns_p - 1, i + w]))
                    aj = int(np.max([0, j - w]))
                    bj = int(np.min([nl_p - 1, j + w]))

                    ind_wind_valid = np.where((valid_index[aj:bj+1, ai:bi+1]).ravel() == 1)
                    position_cand = idlwrap.intarr((bi-ai+1)*(bj-aj+1)) + 1
                    row_wind = row_index[aj:bj+1, ai:bi+1]
                    col_wind = col_index[aj:bj+1, ai:bi+1]

                    for ipair in [0, 1]:
                        for iband in range(0, nb):
                            cand_band = idlwrap.intarr((bi-ai+1)*(bj-aj+1))
                            if ipair == 0:
                                S_S = np.abs(fine1[iband, aj:bj+1, ai:bi+1] - fine1[iband, j, i])
                            else:
                                S_S = np.abs(fine2[iband, aj:bj+1, ai:bi+1] - fine2[iband, j, i])
                            ind_cand = np.where(S_S.ravel() < similar_th[ipair, iband])
                            cand_band[ind_cand] = 1
                            position_cand = position_cand * cand_band

                    indcand = np.where((position_cand != 0) & ((valid_index[aj:bj+1, ai:bi+1]).ravel() == 1))
                    number_cand = len(indcand[0])

                    if number_cand > 5:
                        x_cand = (col_wind.ravel())[indcand]
                        y_cand = (row_wind.ravel())[indcand]
                        finecand = np.zeros((nb*2, number_cand)).astype(float)
                        coarsecand = np.zeros((nb*2, number_cand)).astype(float)

                        for ib in range(0, nb):
                            finecand[ib, :] = (fine1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                            finecand[ib+nb, :] = (fine2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                            coarsecand[ib, :] = (coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                            coarsecand[ib+nb, :] = (coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]

                        if nb == 1:
                            S_D_cand = 1.0 - 0.5*(np.abs((finecand[0, :]-coarsecand[0, :]) / (finecand[0, :]+coarsecand[0, :]+1e-6)) +
                                                  np.abs((finecand[1, :]-coarsecand[1, :]) / (finecand[1, :]+coarsecand[1, :]+1e-6)))
                        else:
                            sdx = np.std(finecand, axis=0, ddof=1)
                            sdy = np.std(coarsecand, axis=0, ddof=1)
                            meanx = np.mean(finecand, axis=0)
                            meany = np.mean(coarsecand, axis=0)
                            x_meanx = finecand - meanx
                            y_meany = coarsecand - meany
                            S_D_cand = nb*2.0*np.mean(x_meanx*y_meany, axis=0) / (sdx*sdy+1e-6) / (nb*2.0-1)

                        S_D_cand[np.isnan(S_D_cand)] = 0.5

                        D_D_cand = np.zeros(number_cand).astype(float)
                        if (bi-ai+1)*(bj-aj+1) < (w*2.0+1)*(w*2.0+1):
                            D_D_cand = 1.0 + np.sqrt((i-x_cand)**2+(j-y_cand)**2) / w
                        else:
                            D_D_cand[0:number_cand] = D_D_all[indcand]

                        C_D = (1.0-S_D_cand) * D_D_cand + 0.0000001
                        weight = (1.0/C_D)/np.sum(1.0/C_D)

                        for ib in range(0, nb):
                            fine_cand = np.hstack(((fine1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand], (fine2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]))
                            coarse_cand = np.hstack(((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand], (coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]))
                            coarse_change = np.abs(np.mean((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]) - np.mean((coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]))
                            
                            V_cand = 1.0
                            if coarse_change >= DN_max*0.02:
                                try:
                                    slope, p_value = analytic_ols_slope_with_pvalue(
                                        coarse_cand,
                                        fine_cand,
                                    )
                                    if p_value <= 0.05 and 0 < slope <= 5:
                                        V_cand = slope
                                except (FloatingPointError, ValueError, TypeError):
                                    pass

                            difc_pair1 = np.abs(np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])-np.mean((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]))+0.01**5
                            difc_pair2 = np.abs(np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])-np.mean((coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]))+0.01**5
                            T_weight1 = (1.0/difc_pair1) / (1.0/difc_pair1+1.0/difc_pair2)
                            T_weight2 = (1.0/difc_pair2) / (1.0/difc_pair1+1.0/difc_pair2)

                            coase0_cand = (coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                            coase1_cand = (coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                            fine01 = fine1[ib, j, i] + np.sum(weight * V_cand * (coase0_cand-coase1_cand))
                            coase2_cand = (coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                            fine02 = fine2[ib, j, i] + np.sum(weight * V_cand * (coase0_cand-coase2_cand))
                            
                            fine0[ib, j, i] = T_weight1 * fine01 + T_weight2 * fine02
                            
                            if fine0[ib, j, i] <= DN_min or fine0[ib, j, i] >= DN_max:
                                fine01 = np.sum(weight*(fine1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand])
                                fine02 = np.sum(weight*(fine2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand])
                                fine0[ib, j, i] = T_weight1 * fine01 + T_weight2 * fine02

                    else:
                        for ib in range(0, nb):
                            difc_pair1 = np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])-np.mean((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])+0.01**5
                            difc_pair1_a = np.abs(difc_pair1)
                            difc_pair2 = np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])-np.mean((coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])+0.01**5
                            difc_pair2_a = np.abs(difc_pair2)
                            T_weight1 = (1.0/difc_pair1_a) / (1.0/difc_pair1_a+1.0/difc_pair2_a)
                            T_weight2 = (1.0/difc_pair2_a) / (1.0/difc_pair1_a+1.0/difc_pair2_a)
                            fine0[ib, j, i] = T_weight1 * (fine1[ib, j, i] + difc_pair1) + T_weight2 * (fine2[ib, j, i] + difc_pair2)

        Out_Name = os.path.join(temp_file, f'temp_blended{isub + 1}{suffix}')
        writeimage(fine0, Out_Name, path1)

    print("   [ESTARFM] 正在合并影像切块...")
    out_final_path = os.path.join(temp_file, f"ESTARFM_Pred_Final{suffix}")
    
    in_ds = gdal.Open(path1)
    if suffix.lower() == '.tif':
        driver = gdal.GetDriverByName("GTiff")
    else:
        driver = gdal.GetDriverByName("ENVI")
    dataset = driver.Create(out_final_path, orig_ns, orig_nl, nb, gdal.GDT_Float32)

    for isub in range(n_nl * n_ns):
        out_name = os.path.join(temp_file, f'temp_blended{isub+1}{suffix}')
        c1, r1 = ind_patch[isub, 0], ind_patch[isub, 2]
        # 替换为 safe_read_raster
        _, _, datavalue = safe_read_raster(out_name)
        for j in range(nb):
            dataset.GetRasterBand(j + 1).WriteArray(datavalue[j, :, :], int(c1), int(r1))

    dataset.SetGeoTransform(in_ds.GetGeoTransform())
    dataset.SetProjection(in_ds.GetProjection())
    dataset.FlushCache()
    dataset = None

    return out_final_path

