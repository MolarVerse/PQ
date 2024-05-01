#include "selection.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> pq_python::select(
    const std::string &selection,
    const std::string &restartFileName,
    const std::string &moldescriptorFileName
)
{
    PyObject *pName;
    PyObject *pModule;
    PyObject *pDict;
    PyObject *pFunc;
    PyObject *pArgs;
    PyObject *pValue;

    ::Py_Initialize();

    const std::string moduleString = "PQAnalysis.topology.api";

    pName = ::PyUnicode_FromString(moduleString.c_str());

    pModule = ::PyImport_Import(pName);
    if (pModule == nullptr)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", moduleString);
    }
    pDict = ::PyModule_GetDict(pModule);

    pFunc = ::PyDict_GetItemString(pDict, "select_from_restart_file");

    if (PyCallable_Check(pFunc))
        PyObject_CallObject(pFunc, NULL);
    else
        PyErr_Print();

    if (moldescriptorFileName.empty() || ::fopen(moldescriptorFileName.c_str(), "r") == nullptr)
        pArgs = ::PyTuple_New(2);
    else
        pArgs = ::PyTuple_New(3);

    pValue = ::PyUnicode_FromString(selection.c_str());
    ::PyTuple_SetItem(pArgs, 0, pValue);
    pValue = ::PyUnicode_FromString(restartFileName.c_str());
    ::PyTuple_SetItem(pArgs, 1, pValue);

    if (!moldescriptorFileName.empty() && ::fopen(moldescriptorFileName.c_str(), "r") != nullptr)
    {
        pValue = ::PyUnicode_FromString(moldescriptorFileName.c_str());
        ::PyTuple_SetItem(pArgs, 2, pValue);
    }

    // print length of pArgs
    printf("Length of pArgs: %ld\n", ::PyTuple_Size(pArgs));

    pValue = ::PyObject_CallObject(pFunc, pArgs);

    if (pValue == nullptr)
    {
        PyErr_Print();
        fprintf(stderr, "Call failed\n");
    }

    auto *numpy_array = reinterpret_cast<PyArrayObject *>(pValue);
    // check if numpy_array is NULL
    if (numpy_array == nullptr)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load numpy array\n");
    }
    npy_intp size = PyArray_SIZE(numpy_array);
    printf("Result of call: %ld\n", ::PyLong_AsLong(pValue));

    // Get pointer to numpy array data
    int *data = static_cast<int *>(PyArray_DATA(numpy_array));
    printf("Result of call: %ld\n", ::PyLong_AsLong(pValue));

    // Initialize std::vector with numpy array data
    std::vector<int> result(data, data + size);
    printf("Result of call: %ld\n", ::PyLong_AsLong(pValue));

    // Now you can use vec...

    Py_DECREF(pValue);
    Py_DECREF(pArgs);
    Py_DECREF(pModule);
    Py_DECREF(pName);

    ::Py_Finalize();

    return result;
}