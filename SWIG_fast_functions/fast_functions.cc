#include "stdint.h"
#include "stdio.h"

void post_processing(uint8_t* F, int width, int height, int length,
    const uint8_t* S, int width_, int height_, int length_, float threshold, bool top2) {
  int F_sum = 0;
  for (int i = 0; i < width * height * length; i++) {
    F_sum += F[i];
  }
  if (F_sum == 0 or F_sum >= width * height * length / 2) {
    return;
  }
  bool* marked = new bool[width * height * length];
  for (int i = 0; i < width * height * length; i++) {
    marked[i] = false;
  }
  int* queue = new int[F_sum];
  int* volume = new int[F_sum];
  int head = 0;
  int tail = 0;
  int bestHead = 0;
  int bestTail = 0;
  int bestHead2 = 0;
  int bestTail2 = 0;
  for (int i = 0; i < width * height * length; i++) {
    if (F[i] && !marked[i]) {
      int temp = head;
      marked[i] = true;
      queue[tail] = i;
      tail++;
      while (head < tail) {
        int x = queue[head] / (height * length);
        int y = queue[head] % (height * length) / length;
        int z = queue[head] % length;
        if (x > 0) {
          int t = queue[head] - height * length;
          if (F[t] && !marked[t]) {
            marked[t] = true;
            queue[tail] = t;
            tail++;
          }
        }
        if (x < width - 1) {
          int t = queue[head] + height * length;
          if (F[t] && !marked[t]) {
            marked[t] = true;
            queue[tail] = t;
            tail++;
          }
        }
        if (y > 0) {
          int t = queue[head] - length;
          if (F[t] && !marked[t]) {
            marked[t] = true;
            queue[tail] = t;
            tail++;
          }
        }
        if (y < height - 1) {
          int t = queue[head] + length;
          if (F[t] && !marked[t]) {
            marked[t] = true;
            queue[tail] = t;
            tail++;
          }
        }
        if (z > 0) {
          int t = queue[head] - 1;
          if (F[t] && !marked[t]) {
            marked[t] = true;
            queue[tail] = t;
            tail++;
          }
        }
        if (z < length - 1) {
          int t = queue[head] + 1;
          if (F[t] && !marked[t]) {
            marked[t] = true;
            queue[tail] = t;
            tail++;
          }
        }
        head++;
      }
      int volume_ = tail - temp;
      for (int l = temp; l < tail; l++) {
        volume[l] = volume_;
      }
      if (volume_ > bestTail - bestHead) {
        bestHead = temp;
        bestTail = tail;
      } else if (volume_ > bestTail2 - bestHead2) {
        bestHead2 = temp;
        bestTail2 = tail;
      }
    }
  }
  int min_volume = bestTail - bestHead;
  if (top2) {
    min_volume = bestTail2 - bestHead2;
  }
  for (int i = 0; i < width * height * length; i++) {
    F[i] = 0;
  }
  for (int l = 0; l < tail; l++) {
    if (volume[l] >= threshold * min_volume) {
      F[queue[l]] = 1;
    }
  }
  delete marked;
  delete queue;
  delete volume;
}

void DSC_computation(const uint8_t* A, int width1, int height1, int length1,
    const uint8_t* G, int width2, int height2, int length2, uint32_t* P, int count) {
  P[0] = 0;
  P[1] = 0;
  P[2] = 0;
  for (int i = 0; i < width1 * height1 * length1; i++) {
    if (A[i]) {
      P[0]++;
      if (G[i]) {
        P[1]++;
        P[2]++;
      }
    } else if (G[i]) {
      P[1]++;
    }
  }
}
