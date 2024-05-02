#include "selection.hpp"

#include <Python.h>

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

    pFunc = ::PyDict_GetItemString(pDict, "select_from_restart_file_as_list");

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

    if (pFunc == NULL)
        ::fprintf(stderr, "pFunc is NULL\n");

    if (pArgs == NULL)
        ::fprintf(stderr, "pArgs is NULL\n");

    if (!PyCallable_Check(pFunc))
        ::fprintf(stderr, "pFunc is not a callable object\n");

    if (!PyTuple_Check(pArgs))
        ::fprintf(stderr, "pArgs is not a tuple object\n");

    pValue = ::PyObject_CallObject(pFunc, pArgs);

    if (pValue == NULL)
    {
        ::PyErr_Print();
        ::fprintf(stderr, "Call failed\n");
    }

    // convert pValue from python list to C++ vector of int result
    std::vector<int> result;
    if (pValue != nullptr)
    {
        PyObject *pIter = ::PyObject_GetIter(pValue);
        PyObject *pItem;

        while ((pItem = ::PyIter_Next(pIter)))
        {
            result.push_back(::PyLong_AsLong(pItem));
            ::Py_DECREF(pItem);
        }

        ::Py_DECREF(pIter);
    }
    else
        ::PyErr_Print();

    Py_DECREF(pValue);
    Py_DECREF(pArgs);
    Py_DECREF(pModule);
    Py_DECREF(pName);

    ::Py_Finalize();

    return result;
}