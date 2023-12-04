#include <ros.h>
#include <std_msgs/Float64MultiArray.h>

// Define ROS NodeHandle
ros::NodeHandle nh;

// Define variables
const int numReadings = 100;
float readings[numReadings];
int readIndex = 0;
float total = 0;
float current_sensor_value = 0.0;
float desired_current_value = 0.0;
float kp = 0.1;  // Proportional gain
float ki = 0.01; // Integral gain
float integral = 0.0;
float pwm_signal = 0.0;

// Define PWM pins and motor controller settings
const int pwmPin = 9; // Replace with your actual PWM pin
const int currentSensorPin = A0; // Replace with your actual analog pin for current sensor

// Define the PWM range for your motor controller
const float pwmMin = 0.0;
const float pwmMax = 255.0;

// Define the loop rate
const int loop_rate = 100; // Hz

// ROS callback to update desired current value
void currentCallback(const std_msgs::Float64MultiArray& msg) {
  desired_current_value = msg.data[2];
}

// ROS Subscribers
ros::Subscriber<std_msgs::Float64MultiArray> sub("/linear", &currentCallback);

void setup() {
  // Initialize ROS
  nh.initNode();
  
  // Setup PWM pin
  pinMode(pwmPin, OUTPUT);
  
  // Setup ROS Subscribers
  nh.subscribe(sub);

  // Initialize readings to 0
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;
  }
}

float averageReading() {
  total = total - readings[readIndex];
  readings[readIndex] = analogRead(currentSensorPin);
  total = total + readings[readIndex];
  readIndex = (readIndex + 1) % numReadings;
  return total / numReadings;
}

void loop() {
  // Handle ROS communication
  nh.spinOnce();
  
  // Read averaged current sensor value
  current_sensor_value = averageReading();

  // Calculate error
  float error = desired_current_value - current_sensor_value;

  // Update integral term
  integral += error;

  // Calculate PWM signal using PI controller
  pwm_signal = kp * error + ki * integral;

  // Clip PWM signal to stay within the motor controller's range
  pwm_signal = constrain(pwm_signal, pwmMin, pwmMax);

  // Apply PWM signal to the motor controller
  analogWrite(pwmPin, pwm_signal);

  // Delay to control loop rate
  delay(1000 / loop_rate);
}
