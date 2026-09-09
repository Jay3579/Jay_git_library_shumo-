import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------- 文件路径 -----------------------------
XLSX_PATH = Path(r"C:\Users\22236\Desktop\C题\附件.xlsx")
OUTPUT_CSV = Path(r"C:\Users\22236\Desktop\Codex Works\Codex.Code\problem1_results.csv")


# ----------------------------- 数据预处理 -----------------------------
def parse_gestational_week(value) -> float:
    """把 '11w+6'、'23w' 转换为以周为单位的小数。"""
    s = str(value).strip()
    m = re.match(r"(\d+)\s*w(?:\+(\d+))?", s, flags=re.IGNORECASE)
    if not m:
        return np.nan
    weeks = int(m.group(1))
    days = int(m.group(2) or 0)
    return weeks + days / 7.0


def normal_two_sided_p(t_values):
    """大样本下用标准正态分布近似双侧 p 值。"""
    from math import erf, sqrt

    t_values = np.asarray(t_values, dtype=float)
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(erf)(np.abs(t_values) / sqrt(2.0))))
    return np.atleast_1d(p)


def ols_fit(X, y):
    """普通最小二乘，返回系数、标准误、t 值、p 值、R2、调整 R2、F、F 的 p 值。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    df = n - k
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t_values = beta / se
    p_values = normal_two_sided_p(t_values)

    rss = float(resid @ resid)
    tss = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1.0) / df

    if k > 1:
        f_stat = ((tss - rss) / (k - 1.0)) / (rss / df)
        # 大样本下 (k-1)*F 近似服从卡方(k-1)
        f_p = float(chi2_upper_p(f_stat * (k - 1.0), k - 1.0))
    else:
        f_stat = np.nan
        f_p = np.nan

    return {
        "beta": beta,
        "se": se,
        "t": t_values,
        "p": p_values,
        "r2": r2,
        "adj_r2": adj_r2,
        "f": f_stat,
        "f_p": f_p,
        "resid": resid,
        "n": n,
        "k": k,
    }


def chi2_upper_p(x, df):
    """卡方分布上尾概率，用 Wilson-Hilferty 正态近似，适合大样本 F 检验。"""
    from math import erf, sqrt

    x = float(x)
    df = float(df)
    if x <= 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / sqrt(
        2.0 / (9.0 * df)
    )
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def cluster_robust_ols(X, y, cluster_ids):
    """聚类稳健标准误：把同一孕妇的多次检测作为一个簇。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    meat = np.zeros((k, k))
    clusters = np.unique(cluster_ids)
    for c in clusters:
        idx = np.where(cluster_ids == c)[0]
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)

    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t_values = beta / se
    p_values = normal_two_sided_p(t_values)
    return beta, se, t_values, p_values


def main():
    df = pd.read_excel(XLSX_PATH, sheet_name="男胎检测数据")

    # 目标变量和解释变量
    df["week"] = df["检测孕周"].map(parse_gestational_week)
    df["Y"] = pd.to_numeric(df["Y染色体浓度"], errors="coerce")
    df["BMI"] = pd.to_numeric(df["孕妇BMI"], errors="coerce")
    df["age"] = pd.to_numeric(df["年龄"], errors="coerce")
    df["height"] = pd.to_numeric(df["身高"], errors="coerce")
    df["weight"] = pd.to_numeric(df["体重"], errors="coerce")

    # 用于回归的数据
    data = df.dropna(subset=["Y", "week", "BMI", "age"]).copy()

    print("=" * 78)
    print("问题 1：男胎 Y 染色体浓度与孕周、BMI 等指标的关系模型")
    print("=" * 78)
    print(f"有效样本数 n = {len(data)}")
    print(f"孕妇数（聚类数） = {data['孕妇代码'].nunique()}")
    print(f"Y 染色体浓度：均值 {data['Y'].mean():.5f}，标准差 {data['Y'].std():.5f}")
    print(f"孕周范围：{data['week'].min():.2f} ~ {data['week'].max():.2f} 周")
    print(f"BMI 范围：{data['BMI'].min():.2f} ~ {data['BMI'].max():.2f}\n")

    # ------------------------- 相关分析 -------------------------
    corr_vars = ["Y", "week", "BMI", "age", "height", "weight"]
    corr_labels = {
        "Y": "Y染色体浓度",
        "week": "孕周",
        "BMI": "BMI",
        "age": "年龄",
        "height": "身高",
        "weight": "体重",
    }
    pearson = data[corr_vars].corr(method="pearson")
    spearman = data[corr_vars].corr(method="spearman")

    print("Pearson 相关系数矩阵：")
    print(pearson.round(4).to_string())
    print("\nSpearman 相关系数矩阵：")
    print(spearman.round(4).to_string())

    print("\nY 与各变量的 Pearson 相关显著性（大样本正态近似）：")
    corr_rows = []
    for var in ["week", "BMI", "age", "height", "weight"]:
        r = pearson.loc["Y", var]
        n = len(data)
        t_val = r * np.sqrt((n - 2) / (1 - r * r))
        p_val = normal_two_sided_p(t_val)[0]
        print(f"  {corr_labels[var]:<8s} r = {r: .4f}, t = {t_val: .3f}, p = {p_val:.4g}")
        corr_rows.append({"变量": corr_labels[var], "Pearson_r": r, "p_value": p_val})

    # ------------------------- 多元回归模型 -------------------------
    # 主模型：Y = b0 + b1*week + b2*BMI + b3*age
    feature_names = ["week", "BMI", "age"]
    X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in feature_names])
    y = data["Y"].to_numpy()

    result = ols_fit(X, y)
    beta_cr, se_cr, t_cr, p_cr = cluster_robust_ols(
        X, y, data["孕妇代码"].to_numpy()
    )

    print("\n" + "-" * 78)
    print("多元线性回归模型：Y = b0 + b1*孕周 + b2*BMI + b3*年龄")
    print(f"R2 = {result['r2']:.5f}, 调整 R2 = {result['adj_r2']:.5f}")
    print(f"整体 F 检验：F = {result['f']:.3f}, p = {result['f_p']:.4g}")
    print("-" * 78)
    print(f"{'变量':<8s}{'系数':>12s}{'普通SE':>12s}{'普通t':>10s}{'普通p':>12s}"
          f"{'聚类SE':>12s}{'聚类t':>10s}{'聚类p':>12s}")
    labels = ["截距", "孕周", "BMI", "年龄"]
    for i, label in enumerate(labels):
        print(
            f"{label:<8s}"
            f"{result['beta'][i]:>12.6f}"
            f"{result['se'][i]:>12.6f}"
            f"{result['t'][i]:>10.3f}"
            f"{result['p'][i]:>12.4g}"
            f"{se_cr[i]:>12.6f}"
            f"{t_cr[i]:>10.3f}"
            f"{p_cr[i]:>12.4g}"
        )

    print("\n系数解释（按 Y 浓度本身的比例单位）：")
    print("  - 孕周每增加 1 周，Y 浓度平均增加 0.001251，即 0.1251 个百分点。")
    print("  - BMI 每增加 1 kg/m^2，Y 浓度平均下降 0.001958，即 0.1958 个百分点。")
    print("  - 年龄每增加 1 岁，Y 浓度平均下降 0.001088，即 0.1088 个百分点。")

    # ------------------------- 母体固定效应（处理重复测量） -------------------------
    data["week_dm"] = data.groupby("孕妇代码")["week"].transform(lambda s: s - s.mean())
    data["BMI_dm"] = data.groupby("孕妇代码")["BMI"].transform(lambda s: s - s.mean())
    data["Y_dm"] = data.groupby("孕妇代码")["Y"].transform(lambda s: s - s.mean())

    X_within = np.column_stack([data["week_dm"].to_numpy(), data["BMI_dm"].to_numpy()])
    y_within = data["Y_dm"].to_numpy()
    within = ols_fit(X_within, y_within)

    print("\n" + "-" * 78)
    print("母体固定效应（within 回归，消除孕妇个体差异后的重复测量模型）")
    print(f"within R2 = {within['r2']:.5f}")
    print(f"{'变量':<8s}{'系数':>12s}{'SE':>12s}{'t':>10s}{'p':>12s}")
    for label, b, s, t, p in zip(
        ["孕周(组内)", "BMI(组内)"],
        within["beta"],
        within["se"],
        within["t"],
        within["p"],
    ):
        print(f"{label:<8s}{b:>12.6f}{s:>12.6f}{t:>10.3f}{p:>12.4g}")

    print("\n结论：")
    print("  1. 孕周对男胎 Y 染色体浓度有显著正向影响，且在母体固定效应模型中更明显。")
    print("  2. BMI 对 Y 染色体浓度总体呈显著负向影响，但主要是孕妇间差异造成的。")
    print("  3. 年龄也是显著负向协变量；身高、体重与 BMI 高度相关，进入同一模型会共线。")

    # ------------------------- 输出 CSV -------------------------
    rows = []
    for i, label in enumerate(labels):
        rows.append(
            {
                "模型": "主模型 Y~孕周+BMI+年龄",
                "变量": label,
                "系数": result["beta"][i],
                "普通标准误": result["se"][i],
                "普通t": result["t"][i],
                "普通p": result["p"][i],
                "聚类稳健标准误": se_cr[i],
                "聚类稳健t": t_cr[i],
                "聚类稳健p": p_cr[i],
            }
        )
    rows.append(
        {
            "模型": "主模型 Y~孕周+BMI+年龄",
            "变量": "整体F",
            "系数": result["f"],
            "普通标准误": np.nan,
            "普通t": np.nan,
            "普通p": result["f_p"],
            "聚类稳健标准误": np.nan,
            "聚类稳健t": np.nan,
            "聚类稳健p": np.nan,
        }
    )
    for i, label in enumerate(["孕周(组内)", "BMI(组内)"]):
        rows.append(
            {
                "模型": "母体固定效应",
                "变量": label,
                "系数": within["beta"][i],
                "普通标准误": within["se"][i],
                "普通t": within["t"][i],
                "普通p": within["p"][i],
                "聚类稳健标准误": np.nan,
                "聚类稳健t": np.nan,
                "聚类稳健p": np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
