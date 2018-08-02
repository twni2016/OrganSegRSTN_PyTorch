import numpy as np
import fast_functions as ff
import time


def DSC_computation(label, pred):
	pred_sum = pred.sum()
	label_sum = label.sum()
	inter_sum = (pred & label).sum()
	return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum


def post_processing(F, S, threshold, top2):
	F_sum = F.sum()
	if F_sum == 0:
		return F
	if F_sum >= np.product(F.shape) / 2:
		return F
	height  = F.shape[0]
	width = F.shape[1]
	depth = F.shape[2]
	ll = np.array(np.nonzero(S))
	marked = np.zeros(F.shape, dtype = np.bool)
	queue = np.zeros((F_sum, 3), dtype = np.int)
	volume = np.zeros(F_sum, dtype = np.int)
	head = 0
	tail = 0
	bestHead = 0
	bestTail = 0
	bestHead2 = 0
	bestTail2 = 0
	for l in range(ll.shape[1]):
		if not marked[ll[0, l], ll[1, l], ll[2, l]]:
			temp = head
			marked[ll[0, l], ll[1, l], ll[2, l]] = True
			queue[tail, :] = [ll[0, l], ll[1, l], ll[2, l]]
			tail = tail + 1
			while (head < tail):
				t1 = queue[head, 0]
				t2 = queue[head, 1]
				t3 = queue[head, 2]
				if t1 > 0 and F[t1 - 1, t2, t3] and not marked[t1 - 1, t2, t3]:
					marked[t1 - 1, t2, t3] = True
					queue[tail, :] = [t1 - 1, t2, t3]
					tail = tail + 1
				if t1 < height - 1 and F[t1 + 1, t2, t3] and not marked[t1 + 1, t2, t3]:
					marked[t1 + 1, t2, t3] = True
					queue[tail, :] = [t1 + 1, t2, t3]
					tail = tail + 1
				if t2 > 0 and F[t1, t2 - 1, t3] and not marked[t1, t2 - 1, t3]:
					marked[t1, t2 - 1, t3] = True
					queue[tail, :] = [t1, t2 - 1, t3]
					tail = tail + 1
				if t2 < width - 1 and F[t1, t2 + 1, t3] and not marked[t1, t2 + 1, t3]:
					marked[t1, t2 + 1, t3] = True
					queue[tail, :] = [t1, t2 + 1, t3]
					tail = tail + 1
				if t3 > 0 and F[t1, t2, t3 - 1] and not marked[t1, t2, t3 - 1]:
					marked[t1, t2, t3 - 1] = True
					queue[tail, :] = [t1, t2, t3 - 1]
					tail = tail + 1
				if t3 < depth - 1 and F[t1, t2, t3 + 1] and not marked[t1, t2, t3 + 1]:
					marked[t1, t2, t3 + 1] = True
					queue[tail, :] = [t1, t2, t3 + 1]
					tail = tail + 1
				head = head + 1
			if tail - temp > bestTail - bestHead:
				bestHead2 = bestHead
				bestTail2 = bestTail
				bestHead = temp
				bestTail = tail
			elif tail - temp > bestTail2 - bestHead2:
				bestHead2 = temp
				bestTail2 = tail
			volume[temp: tail] = tail - temp
	volume = volume[0: tail]
	if top2:
		target_voxel = np.where(volume >= (bestTail2 - bestHead2) * threshold)
	else:
		target_voxel = np.where(volume >= (bestTail - bestHead) * threshold)
	F0 = np.zeros(F.shape, dtype = np.bool)
	F0[tuple(map(tuple, np.transpose(queue[target_voxel, :])))] = True
	return F0

print('python')
G = np.zeros((512,512,240),dtype=np.uint8)
G[128:384,128:384,60:180]=1
volume_data = np.load('1.npz')
F = volume_data['volume'].astype(np.uint8)
start_time = time.time()
F = post_processing(F, F, 1.0, False)
print(time.time() - start_time)
start_time = time.time()
for l in range(10):
	DSC = DSC_computation(F,G)
print(DSC)
print(time.time() - start_time)


print('SWIG')
volume_data = np.load('1.npz')
G = np.zeros((512,512,240),dtype=np.uint8)
G[128:384,128:384,60:180]=1
F = volume_data['volume'].astype(np.uint8)
start_time = time.time()
ff.post_processing(F, F, 1.0, False)
print(time.time() - start_time)
start_time = time.time()
for l in range(10):
	P = np.zeros(3, dtype = np.uint32)
	ff.DSC_computation(F,G,P)
print(P, float(P[2]) * 2 / (P[0] + P[1]))
print(time.time() - start_time)

