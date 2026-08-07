from utils import collect_metrics

df_sym = collect_metrics("sym")
print("Updated metrics in performance_sym.csv")
print(df_sym)

print()

df_dis = collect_metrics("dis")
print("Updated metrics in performance_dis.csv")
print(df_dis)