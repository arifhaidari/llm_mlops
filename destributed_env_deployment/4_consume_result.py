from .rabbit_mq import RabbitBuffer

buffer = RabbitBuffer("llama-results")

results = buffer.consume(10_000) # we can consume the whole qeuede or just a few of them like 20
# results = buffer.consume(20)
len(results)

print(results[9000].decode())
