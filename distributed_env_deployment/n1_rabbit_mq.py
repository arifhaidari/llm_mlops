# Importing the 'pika' library, which is a Python implementation of the AMQP 0.9.1 protocol for RabbitMQ,
# used for connecting, sending, and receiving messages in a distributed environment.
import pika

# Defining the credentials for RabbitMQ server:
# RABBIT_USER is the username for the RabbitMQ user.
# RABBIT_PASS is the password for authenticating the user.
RABBIT_USER = "rabbitmq_user"
RABBIT_PASS = "pass"

# HEAD_IP is the IP address of the RabbitMQ head node (server).
# RABBIT_PORT is the port where RabbitMQ is running, typically 5672 for RabbitMQ's AMQP protocol.
HEAD_IP = "your_head_node_ip"
RABBIT_PORT = 5672


class RabbitBuffer:
     """
     RabbitBuffer class handles communication with RabbitMQ for message queueing.
     It allows producing (sending) and consuming (receiving) messages to/from a specified queue.
     """

     def __init__(self, queue_name: str):
          """
          Initializes a connection to RabbitMQ and declares the queue.

          Parameters:
          queue_name (str): The name of the queue where messages will be sent or consumed.
          """
          # Storing the name of the queue.
          self.queue_name = queue_name

          # Creating a PlainCredentials object using the RabbitMQ username and password.
          # This is used for authenticating the connection.
          self.credentials = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)

          # Establishing a connection to RabbitMQ using BlockingConnection.
          # ConnectionParameters takes the head node IP, port, virtual host ("/"), and credentials for authentication.
          self.connection = pika.BlockingConnection(pika.ConnectionParameters(HEAD_IP, RABBIT_PORT, "/", self.credentials))

          # Creating a channel over the established connection.
          # The channel is a communication path to RabbitMQ over which we can send/receive messages.
          self.channel = self.connection.channel()

          # Declaring a queue with the specified queue name.
          # durable=True ensures that the queue and its messages persist even if RabbitMQ is restarted.
          self.queue = self.channel.queue_declare(queue=self.queue_name, durable=True)

     def produce(self, messages: list[str]):
          """
          Publishes messages to the RabbitMQ queue.

          Parameters:
          messages (list[str]): A list of messages (strings) to be sent to the queue.
          """
          # Iterate through each message in the list and publish it to the queue.
          for message in messages:
               self.channel.basic_publish(
                    exchange="",  # No exchange used, sending directly to the queue.
                    routing_key=self.queue_name,  # The queue name serves as the routing key.
                    body=message,  # The message body is the string message to be sent.
                    # Properties define how the message should be handled.
                    # delivery_mode=2 makes the message persistent, ensuring it's stored on disk and not lost if RabbitMQ restarts.
                    properties=pika.BasicProperties(delivery_mode=2),
               )

     def consume(self, num_messages: int):
          """
          Consumes (retrieves) a specific number of messages from the queue.

          Parameters:
          num_messages (int): The number of messages to retrieve from the queue.

          Returns:
          list: A list of messages retrieved from the queue.
          """
          # List to hold the retrieved messages.
          messages = []

          # Retrieve messages one at a time, up to num_messages.
          for _ in range(num_messages):
               # Fetch a single message from the queue using basic_get, which is a synchronous method.
               # It returns a method frame, header frame, and the message body.
               method_frame, header_frame, body = self.channel.basic_get(queue=self.queue_name)

               # If a message was successfully retrieved (i.e., method_frame is not None):
               if method_frame:
                    # Append the message body to the list of messages.
                    messages.append(body)

                    # Acknowledge that the message has been processed, removing it from the queue.
                    # This prevents the message from being re-queued and sent again.
                    self.channel.basic_ack(method_frame.delivery_tag)

          # Return the list of retrieved messages.
          return messages


"""
What is pika?
pika is a Python library that provides tools for working with RabbitMQ, which is a message broker 
that implements the Advanced Message Queuing Protocol (AMQP).

RabbitMQ is widely used in distributed systems to manage queues, send/receive messages between 
components, and decouple the processes or services. It allows for asynchronous communication, 
which is essential in distributed environments.

pika facilitates:

Connections to RabbitMQ: Helps in connecting to RabbitMQ servers and authenticating with credentials.
Channels: Creates virtual communication lines (channels) to RabbitMQ for message publishing and consumption.
Message Publishing and Consumption: Allows sending messages to queues (produce method) and 
retrieving them (consume method).
Persistence: With message persistence, it ensures that messages are not lost even if RabbitMQ crashes.
How pika Helps in Distributed Environment:
Decoupling Components: Services in a distributed environment can communicate through RabbitMQ 
without being directly connected.
Fault Tolerance: Persistent queues and messages ensure that data isn't lost, even during crashes 
or restarts.
Scalability: RabbitMQ can distribute messages across multiple consumers, enabling horizontal 
scaling of services.
In this context, pika enables reliable communication between the language model deployment 
components in a distributed system, ensuring that generated results or requests can be queued, 
distributed, and processed efficiently.

"""
