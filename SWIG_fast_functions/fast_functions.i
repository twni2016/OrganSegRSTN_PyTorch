%module fast_functions
%{
  #define SWIG_FILE_WITH_INIT
  #include "fast_functions.h"
%}

%include "numpy.i"
%include "stdint.i"

%init %{
  import_array();
%}

%apply (uint8_t* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
  {(uint8_t* F, int width, int height, int length)};

%apply (uint8_t* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
  {(const uint8_t* S, int width_, int height_, int length_)};

%apply (uint8_t* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
  {(const uint8_t* A, int width1, int height1, int length1)};

%apply (uint8_t* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
  {(const uint8_t* G, int width2, int height2, int length2)};

%apply (uint32_t* INPLACE_ARRAY1, int DIM1)
  {(uint32_t* P, int count)};

%include "fast_functions.h"
