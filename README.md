# Inverse_Modeling_of_EMA


## Experimental Study on Microparticle Trajectory-Tracking

Embarking on an intriguing journey through the dynamic world of microparticle control, this experimental odyssey delves into the mesmerizing dance of a N42 permanent magnet sphere, merely 0.5 mm in radius. Enveloped in the embrace of a high-viscosity fluid realm, mirroring the delicate choreography of medical procedures within bronchial landscapes (inspired by the work of Samuel K. Lai, Ying-Ying Wang, Denis Wirtz, 2010), our experiment unfolds in the midst of Newtonian liquid currents, navigating the complex nuances of fluid dynamics. As we venture into uncharted territories, our stage is set with a vivid palette of viscous intrigue, where every interaction tells a tale, and the fluidic dance unfolds with a frictional ballet, guided by the mysterious coefficient of 0.377 Ns/m.

## Experimental Setup

The experimentation process involves three main phases:

1. **Training Models**
The training process involves the use of a Random Forest Classifier to model the relationship between input features and output labels. 
* Data Loading and Preprocessing:
The EmaRegressor class is defined to encapsulate the training and evaluation process.
Data is loaded from a specified file, and preprocessing steps include encoding labels and scaling features.
# Relevant part from random_forest_classifier_*EMA.py

   ```import matplotlib.pyplot as plt
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import mean_squared_error, mean_absolute_error
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import MinMaxScaler, LabelEncoder
   import numpy as np
   import joblib

   class EmaRegressor:
      # Constructor initializes instance variables and loads data from a file.
      def __init__(self, data_path='ema_data/*EMA/*EMA.txt'):
         # Load data from the specified file path.
         self.data = self.load_data(data_path)
         # LabelEncoder and MinMaxScaler for data preprocessing.
         self.le = LabelEncoder()
         self.scaler = MinMaxScaler()
         # Lists to store errors, out-of-bag scores, and other metrics during training.
         self.error = []
         self.oob = []
         self.error_test = []
         self.n_est = []
   ```
* Model Training and Evaluation:
The train_model method trains a Random Forest Classifier with specified hyperparameters.
The evaluate_model method assesses the trained model's performance using Mean Absolute Error, Mean Squared Error, and Out-of-Bag (OOB) error.
The train_and_evaluate_models method iterates over different values of n_estimators, trains a model for each value, evaluates its performance, and stores the results.
Finally, the plot_results method is used to visualize the evaluation results.

    ```# Train and evaluate models for different values of n_estimators.
    def train_and_evaluate_models(self, n_estimators_values):
        for i, n_estimators in enumerate(n_estimators_values, start=1):
            x_train, y_train, _ = self.preprocess_data(1.0 * i)
            x_test, y_test, _ = self.preprocess_data(1.0 * i)

            model = self.train_model(x_train, y_train, n_estimators)
            mae, mse, oob_error = self.evaluate_model(model, x_test, y_test)

            # Store metrics and save the trained model to a file.
            self.error.append(mae)
            self.oob.append(oob_error)
            self.error_test.append(mse)
            self.n_est.append(n_estimators)

            filename = f'emaone_{i}.sav'
            joblib.dump(model, filename)
   
   ```
* Main Execution:
The script is executed, creating an instance of the EmaRegressor class, training and evaluating models, and plotting the results.


   ```
   if __name__ == "__main__":
      ema_regressor = EmaRegressor()
      ema_regressor.train_and_evaluate_models(range(10, 110, 10))
      ema_regressor.plot_results()
      plt.show()
   ```

   - `random_forest_classifier_sEMA.py`: Generates models for single EMA.
   - `random_forest_classifier_dEMA.py`: Generates models for double EMA.
   - `random_forest_classifier_qEMA.py`: Generates models for quadruple EMA.
   - Data is stored in the `ema_data` directory.

2. **Model Loading and Application**
   - `random_forest_classifier_model_reader.py`: Loads the trained models stored in `.sav` format.
   - Results are saved to an Excel file for further analysis.

* Initialization:

The class is initialized with parameters related to model file naming conventions and the path to the data file.
It loads the data from the specified file into a NumPy array.
Data Loading:

The load_data method reads data from a text file, organizes it into triplets, and reshapes it into a 2D NumPy array.
* Data Preprocessing:

The preprocess_data method takes a specific column (x_val[:, 2:3]) from the input data, finds the closest value in a predefined mean array, and returns the corresponding index.
* Model Loading:

The load_model method constructs the filename based on the index obtained from preprocessing and loads the corresponding machine learning model.
* Prediction:

The make_prediction method uses the loaded model to predict output values based on the input data.
* Overall Prediction:

The predict method combines the above functionalities to predict outputs for a given set of input values.
* Example Usage:

In the script's main block, an instance of EmaModelReader is created, and an example input x_val is defined.
The predict method is then called to obtain predictions for current_rand and d_rand, which are printed.
Note: The script assumes that pre-trained models exist with filenames following a specific pattern (e.g., 'ematwoexp1.sav', 'ematwoexp2.sav', etc.). Also, make sure to have the required libraries installed (matplotlib, scikit-learn, numpy, joblib).

3. **ROS Noetic Integration**
   - `desired_value_publisher.py`: Publishes trajectory-tracking results on the ROS topic `/linear`.
   - The published results are then sent to the hardware-based controller.

* Initialization:

The class DynamixelController is initialized with the path to an Excel file containing position and current data.
The data is loaded from the Excel file into a Pandas DataFrame (self.data).
A publisher for the "/linear" topic is created.
* Data Retrieval:

The get_next_data_point method retrieves the next data point (position and current) from the loaded Excel data. If there is no more data, it returns None.
* Publishing on "/linear" Topic:

The publish_linear_topic method retrieves the next data point using get_next_data_point.
If there is more data, it creates a Float64MultiArray message with the position and current values and publishes it on the "/linear" topic using the created publisher.
If there is no more data, it logs an info message.
* Main Block:

The script initializes the ROS node.
An instance of DynamixelController is created.
A loop runs continuously, publishing data on the "/linear" topic at a rate of 1 Hz.
* Note:

Replace 'your_excel_file.xlsx' with the actual path to your Excel file.
The script assumes that the Excel file has columns named 'Position' and 'Current' for position and current values.
Ensure that your ROS environment is set up correctly, and the necessary dependencies are installed.
Adjust other parts of your code as needed for specific requirements or integration with other ROS nodes.

4. **Hardware-Based Control**

a. **Position values are sent to `position_controller.py` and `position_controller2.py`, controlling Dynamixel motors**

* Initialization:

The class DynamixelController is initialized with ROS node initialization, setting up parameters, and defining variables for desired motor positions.
* Service and Subscribers Setup:

The script waits for the '/dynamixel_workbench/dynamixel_command' service to be available.
Two subscribers are set up: one for receiving motor state information (/dynamixel_workbench/dynamixel_state) and another for receiving desired motor positions (/linear).
* Callback Functions:

The callback function is called when motor state information is received. It extracts the present position of the motor and calls the ema function.
The callback2 function is called when desired motor positions are received. It updates the desired_value1 and desired_value2 variables.
* EMA (Exponential Moving Average) Calculation:

The ema function calculates the total value by adding the desired value (desired_value1) to the present position of the motor. It then clips the total value to be within the range of -20000 to 20000.
The script uses the '/dynamixel_workbench/dynamixel_command' service to set the goal position of the motor based on the calculated total value.
* Main Block:

In the script's main block, an instance of DynamixelController is created, and the feed function is called to keep the ROS node running.
Note: The script assumes the presence of the Dynamixel motor controller service (/dynamixel_workbench/dynamixel_command) and subscribes to motor state information (/dynamixel_workbench/dynamixel_state) and desired motor positions (/linear). Ensure that the required ROS packages and topics are correctly configured and available.

b. **Current control is implemented using the Arduino-based `current_controller`.**
* Initialization:The script initializes the ROS NodeHandle (nh).
It defines various variables, including the number of readings, PWM pins, motor controller settings, and loop rate.

* ROS Callback:The currentCallback function is called when a message of type std_msgs/Float64MultiArray is received on the "/linear" topic. It updates the desired_current_value based on the third element of the received array.
* ROS Subscriber Setup:The script sets up a ROS subscriber (sub) for the "/linear" topic, linking it to the currentCallback function.

* Setup Function:The setup function initializes the ROS node, sets up the PWM pin as an output, and subscribes to the ROS topic.

* Average Reading Function: The averageReading function calculates the moving average of the analog readings from the current sensor.

* Main Loop: The loop function is the main execution loop.
It handles ROS communication using nh.spinOnce() to process incoming messages.
The averaged current sensor value is obtained using the averageReading function.
The PI controller calculates the PWM signal based on the error between the desired and actual current values.
The PWM signal is constrained to stay within the motor controller's range.
The final PWM signal is applied to the motor controller using analogWrite.
There's a delay to control the loop rate.
Note: Make sure to replace placeholder comments like // Replace with your actual PWM pin with the appropriate pin numbers based on your hardware configuration. Additionally, ensure that the current sensor values, desired current values, and PWM ranges are suitable for your specific setup.

c. **The Kuka robot manipulator is controlled using the [KukaRosOpenCommunication](https://github.com/AytacKahveci/KukaRosOpenCommunication) repository.**

## Dependencies
Ensure the following dependencies are installed:
1. [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
2. [Dynamixel Workbench](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_workbench/)
3. [Arduino](https://docs.arduino.cc/software/ide-v1/tutorials/Linux#toc2)
4. [ROS_packages](https://wiki.ros.org/rosserial_arduino/Tutorials)

## Note:
Because of the github recommended maximum file size of 50.00 MB i could not add the sEMA,dEMA and qEMA data. 
