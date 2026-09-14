import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 获取沪深300指数数据 ==========
print("正在获取沪深300数据...")
df = ak.stock_zh_index_daily(symbol="sh000300")
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)  # 只取近几年的数据，速度快

# ========== 2. 构建因子 ==========
df['ret'] = df['close'].pct_change()                  # 日收益率
df['mom_20'] = df['close'].pct_change(20)              # 20日动量因子
df['vol_20'] = df['ret'].rolling(20).std()             # 20日波动率因子

# 标准化（Z-score）
df['mom_z'] = (df['mom_20'] - df['mom_20'].mean()) / df['mom_20'].std()
df['vol_z'] = (df['vol_20'] - df['vol_20'].mean()) / df['vol_20'].std()

df = df.dropna().reset_index(drop=True)

# ========== 3. 多因子合成信号 ==========
# 动量正向，波动率负向：分数越高越倾向于持有
df['score'] = df['mom_z'] - df['vol_z']

# 月度调仓：每月第一个交易日看信号，决定当月持仓
df['month'] = df['date'].dt.to_period('M')
df['signal'] = np.where(df['score'] > 0, 1, 0)  # 分数大于0持有，否则空仓

# 计算策略收益（按信号持仓）
df['strategy_ret'] = df['signal'].shift(1) * df['ret']
df['strategy_ret'] = df['strategy_ret'].fillna(0)

# ========== 4. 计算绩效指标 ==========
# 累计收益
df['cum_bench'] = (1 + df['ret']).cumprod()
df['cum_strategy'] = (1 + df['strategy_ret']).cumprod()

# 年化收益
days = len(df)
annual_ret = (df['cum_strategy'].iloc[-1]) ** (252 / days) - 1
bench_ret = (df['cum_bench'].iloc[-1]) ** (252 / days) - 1

# 最大回撤
def max_drawdown(cum):
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

mdd = max_drawdown(df['cum_strategy'])

# 夏普比率（无风险利率取0）
sharpe = df['strategy_ret'].mean() / df['strategy_ret'].std() * np.sqrt(252)

print(f"策略年化收益：{annual_ret:.2%}")
print(f"基准年化收益：{bench_ret:.2%}")
print(f"最大回撤：{mdd:.2%}")
print(f"夏普比率：{sharpe:.2f}")

# ========== 5. 可视化 ==========
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['cum_bench'], label='沪深300', color='gray')
plt.plot(df['date'], df['cum_strategy'], label='多因子策略', color='red')
plt.title('多因子策略 vs 沪深300 累计收益对比')
plt.xlabel('日期')
plt.ylabel('累计净值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("backtest_result.png", dpi=150)
plt.show()

print("回测结果图已保存为 backtest_result.png")