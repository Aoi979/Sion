#include <Python.h>

extern "C" PyObject *PyInit__C() {
  static PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C",
      "Sion PyTorch custom operator registration module",
      -1,
      nullptr,
  };
  return PyModule_Create(&module_def);
}
