void post_processing(uint8_t* F, int width, int height, int length,
    const uint8_t* S, int width_, int height_, int length_, float threshold, bool top2);

void DSC_computation(const uint8_t* A, int width1, int height1, int length1,
    const uint8_t* G, int width2, int height2, int length2, uint32_t* P, int count);
