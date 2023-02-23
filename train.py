from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import boto3

import bentoml


if __name__ == "__main__":
    # prepare data
    s3 = boto3.client('s3',
        endpoint_url='https://minio.lab.sspcloud.fr/',
                     )
    data = s3.get_object(Bucket="mbenxsalha", Key="diffusion/yatai_data.csv")["Body"]
    df = pd.read_csv(data)
    features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    x, y = df[features], df['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    
    # train model
    model = RandomForestRegressor(max_depth=8, random_state=0)
    model.fit(x_train, y_train)
    
    # evaluate model
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # save model
    metadata = {
        "RMSE": rmse
    }
    bento_model = bentoml.sklearn.save_model(
        "regressor",
        model, 
        metadata=metadata)
    print(f"Model saved: {bento_model}")
