from keras.callbacks import LambdaCallback
from shakespeare_utils import *

model.summary()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

generate_output()


#from keras.models import Model