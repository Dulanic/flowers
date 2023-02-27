import tensorflow as tf

# Print the list of available GPUs
print("List of available GPUs:")
print(tf.config.list_physical_devices('GPU'))

# Create a TensorFlow session
sess = tf.compat.v1.Session()

# Create a simple TensorFlow operation and add it to the session graph
a = tf.constant(1.0)
b = tf.constant(2.0)
c = a + b

# Run the operation and print the result
result = sess.run(c)
print("Result: ", result)

# Close the TensorFlow session
sess.close()