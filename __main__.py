from network import HopfieldNetwork
import time

if __name__ == '__main__':
    net = HopfieldNetwork(1600)
    for i in range(2):
        arr = net.get_random_pattern()
        net.save_pattern(arr)
    net.train()
    print("finished training")
    start = time.time()
    net.update(100000)
    print("duration", int(time.time()-start))
    print("finished updating")
    print("is_saved:", net.is_saved(net.state))
    net.visualize()