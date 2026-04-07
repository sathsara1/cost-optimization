import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove prefix like "Cloud_Dataset.csv/"
    df.columns = [str(col).split("/")[-1] for col in df.columns]
    return df


def prepare_data(df, num_tasks=30):
    df = clean_columns(df)

    required = [
        "cpu_usage", "memory_usage",
        "vm_type", "vCPU", "RAM_GB", "price_per_hour"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna()
    df = df[(df['cpu_usage'] > 0) & (df['memory_usage'] > 0)]

    tasks_df = df[['cpu_usage', 'memory_usage']].copy()

    tasks_df.rename(columns={
        'cpu_usage': 'cpu',
        'memory_usage': 'ram'
    }, inplace=True)

    # Normalize 
    tasks_df['cpu'] = (tasks_df['cpu'] / tasks_df['cpu'].max() * 4).round().astype(int)
    tasks_df['ram'] = (tasks_df['ram'] / tasks_df['ram'].max() * 8).round().astype(int)

    servers_df = df[['vm_type', 'vCPU', 'RAM_GB', 'price_per_hour']].drop_duplicates()

    servers_df.rename(columns={
        'vCPU': 'cpu',
        'RAM_GB': 'ram',
        'price_per_hour': 'cost'
    }, inplace=True)

    servers_df = servers_df.groupby('vm_type').first().reset_index()

    # Normalize cost 
    servers_df['cost'] = (servers_df['cost'] / servers_df['cost'].max()) * 10

    max_cpu = tasks_df['cpu'].max()
    max_ram = tasks_df['ram'].max()

    servers_df['cpu'] = servers_df['cpu'].clip(lower=max_cpu)
    servers_df['ram'] = servers_df['ram'].clip(lower=max_ram)

    tasks = tasks_df.sample(num_tasks, random_state=42).to_dict('records')
    servers = servers_df[['cpu', 'ram', 'cost']].to_dict('records')

    if len(tasks) == 0:
        raise ValueError("No valid tasks found")

    if len(servers) == 0:
        raise ValueError("No servers found")

    return tasks, servers