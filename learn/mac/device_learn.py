from tinygrad import Device

if __name__ == "__main__":
    print(Device.default) # actual device runtime object
    print(Device.DEFAULT) # selected default device name
