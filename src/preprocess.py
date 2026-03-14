import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

PROTOCOL_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2}
FLAG_MAP = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTO': 3, 'RSTR': 4,
            'S1': 5, 'S2': 6, 'S3': 7, 'OTH': 8, 'SH': 9, 'RSTOS0': 10}


def encode_categoricals(df):
    df['protocol_type'] = df['protocol_type'].map(PROTOCOL_MAP).fillna(3).astype(int)
    df['service'] = pd.Categorical(df['service']).codes
    df['flag'] = df['flag'].map(FLAG_MAP).fillna(0).astype(int)
    return df


def extract_features(df):
    X = pd.DataFrame()
    X['ip_hash']           = df['src_bytes'] % 100000
    X['endpoint_hash']     = df['dst_bytes'] % 10000
    X['http_method']       = df['protocol_type']
    X['payload_size']      = df['src_bytes']
    X['hour_of_day']       = df['duration'] % 24
    X['is_weekend']        = 0
    X['query_param_count'] = df['num_failed_logins']
    X['header_count']      = df['count']
    # cast to a wider integer type to avoid overflow when applying modulo
    X['user_agent_hash']   = df['service'].astype('int32') % 10000
    X['content_type_hash'] = df['flag'] % 1000
    return X.astype(np.float32)


def load_nslkdd(train_path, test_path):
    print("Loading NSL-KDD dataset...")
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=COLUMNS)

    for df in [train_df, test_df]:
        encode_categoricals(df)

    X_train = extract_features(train_df)
    X_test  = extract_features(test_df)

    y_train = (train_df['label'] != 'normal').astype(np.float32).values
    y_test  = (test_df['label']  != 'normal').astype(np.float32).values

    print(f"Train samples: {len(X_train)} | Attacks: {int(y_train.sum())} | Normal: {int((y_train==0).sum())}")
    print(f"Test  samples: {len(X_test)}  | Attacks: {int(y_test.sum())}  | Normal: {int((y_test==0).sum())}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')

    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')

    X_train, X_test, y_train, y_test = load_nslkdd(train_path, test_path)

    print("Preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")