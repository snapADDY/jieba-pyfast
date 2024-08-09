#include <Python.h>
#include <math.h>
#include <float.h>

// Function declarations
int _get_dag_and_calc(PyObject* freqs, PyObject* text, PyObject *route, double total);
PyObject* _viterbi(PyObject* obs, PyObject* _states, PyObject* start_p, PyObject* trans_p, PyObject* emip_p);

// Function implementations
int get_dag(PyObject *dag, PyObject *freqs, PyObject *text)
{
    Py_ssize_t n = PySequence_Size(text);
    PyObject *tmp_list, *frag;
    Py_ssize_t i, k;

    if (n < 0)
        return -1; // handle error: invalid sequence size

    for (k = 0; k < n; k++)
    {
        tmp_list = PyList_New(0);
        if (!tmp_list)
            return -1; // handle error: out of memory

        for (i = k; i < n; i++)
        {
            frag = PyUnicode_Substring(text, k, i + 1);
            if (!frag)
            {
                Py_DECREF(tmp_list);
                return -1; // handle error: failed to get substring
            }

            PyObject *freq_value = PyDict_GetItemWithError(freqs, frag);
            Py_DECREF(frag); // release reference to substring

            if (!freq_value)
            {
                if (PyErr_Occurred())
                {
                    Py_DECREF(tmp_list);
                    return -1; // handle error: dictionary access failure
                }
                break; // stop extending the fragment if it is not in freqs
            }

            if (PyLong_AsLong(freq_value))
            {
                if (PyList_Append(tmp_list, PyLong_FromSsize_t(i)) < 0)
                {
                    Py_DECREF(tmp_list);
                    return -1; // handle error: failed to append to list
                }
            }
        }

        if (PyList_Size(tmp_list) == 0)
        {
            if (PyList_Append(tmp_list, PyLong_FromSsize_t(k)) < 0)
            {
                Py_DECREF(tmp_list);
                return -1; // handle error: failed to append to list
            }
        }

        if (PyDict_SetItem(dag, PyLong_FromSsize_t(k), tmp_list) < 0)
        {
            Py_DECREF(tmp_list);
            return -1; // handle error: failed to set dictionary item
        }
        Py_DECREF(tmp_list); // release reference to tmp_list
    }

    return 1;
}

int _get_dag_and_calc(PyObject* freqs, PyObject* text, PyObject *route, double total)
{
    const Py_ssize_t N = PySequence_Size(text);
    if (N < 0)
        return -1; // handle error: invalid sequence size

    Py_ssize_t (*dag)[20] = malloc(sizeof(Py_ssize_t) * 20 * N);
    Py_ssize_t *points = malloc(sizeof(Py_ssize_t) * N);
    if (!dag || !points)
    {
        free(dag);
        free(points);
        return -1; // handle error: out of memory
    }

    double (*_route)[2] = malloc(sizeof(double) * 2 * (N + 1));
    if (!_route)
    {
        free(dag);
        free(points);
        return -1; // handle error: out of memory
    }

    double logtotal = log(total);
    double max_freq = -DBL_MAX;

    _route[N][0] = 0;
    _route[N][1] = 0;

    for (Py_ssize_t i = 0; i < N; i++)
        points[i] = 0;

    for (Py_ssize_t k = 0; k < N; k++)
    {
        Py_ssize_t i = k;
        PyObject *frag = PySequence_GetItem(text, k);
        while (i < N && (points[k] < 12))
        {
            PyObject *t_f = PyDict_GetItem(freqs, frag);
            if (t_f && PyLong_AsLong(t_f))
            {
                dag[k][points[k]++] = i;
            }
            i += 1;
            Py_XDECREF(frag);
            frag = PySequence_GetSlice(text, k, i + 1);
        }
        Py_XDECREF(frag);
        if (points[k] == 0)
        {
            dag[k][0] = k;
            points[k] = 1;
        }
    }

    for (Py_ssize_t idx = N - 1; idx >= 0; idx--)
    {
        max_freq = -DBL_MAX;
        Py_ssize_t max_x = 0;
        Py_ssize_t t_list_len = points[idx];
        for (Py_ssize_t i = 0; i < t_list_len; i++)
        {
            Py_ssize_t x = dag[idx][i];
            PyObject *slice_of_sentence = PySequence_GetSlice(text, idx, x + 1);
            PyObject *o_freq = PyDict_GetItem(freqs, slice_of_sentence);
            Py_ssize_t fq = o_freq ? PyLong_AsLong(o_freq) : 1;
            fq = (fq == 0) ? 1 : fq;
            Py_XDECREF(slice_of_sentence);

            double fq_2 = _route[x + 1][0];
            double fq_last = log((double)fq) - logtotal + fq_2;
            if (fq_last >= max_freq)
            {
                max_freq = fq_last;
                max_x = x;
            }
        }
        _route[idx][0] = max_freq;
        _route[idx][1] = (double)max_x;
    }

    for (Py_ssize_t i = 0; i <= N; i++)
    {
        if (PyList_Append(route, PyLong_FromSsize_t((Py_ssize_t)_route[i][1])) < 0)
        {
            free(dag);
            free(points);
            free(_route);
            return -1; // handle error: failed to append to list
        }
    }

    free(dag);
    free(points);
    free(_route);
    return 1;
}

#define MIN_FLOAT -3.14e100

PyObject* _viterbi(PyObject* obs, PyObject* _states, PyObject* start_p, PyObject* trans_p, PyObject* emip_p)
{
    const Py_ssize_t obs_len = PySequence_Size(obs);
    if (obs_len < 0)
        return NULL; // handle error: invalid sequence size

    const int states_num = 4;
    PyObject *item, *t_dict, *t_obs, *res_tuple, *t_list, *ttemp;
    Py_ssize_t i, j;
    double t_double, t_double_2, em_p, max_prob, prob;
    double (*V)[22] = malloc(sizeof(double) * obs_len * 22);
    char (*path)[22] = malloc(sizeof(char) * obs_len * 22);
    if (!V || !path)
    {
        free(V);
        free(path);
        return NULL; // handle error: out of memory
    }

    const char* PrevStatus_str[22] = { "ES", "MB", "SE", "BM" };
    char* states = PyUnicode_AsUTF8AndSize(_states, NULL);
    char y, best_state, y0, now_state;
    int p;

    PyObject* emip_p_dict[4];
    PyObject* trans_p_dict[22][2];
    PyObject* py_states[4];

    for (i = 0; i < states_num; i++)
        py_states[i] = PyUnicode_FromStringAndSize(states + i, 1);

    emip_p_dict[0] = PyDict_GetItem(emip_p, py_states[0]);
    emip_p_dict[1] = PyDict_GetItem(emip_p, py_states[1]);
    emip_p_dict[2] = PyDict_GetItem(emip_p, py_states[2]);
    emip_p_dict[3] = PyDict_GetItem(emip_p, py_states[3]);

    trans_p_dict['B' - 'B'][0] = PyDict_GetItem(trans_p, py_states[2]);
    trans_p_dict['B' - 'B'][1] = PyDict_GetItem(trans_p, py_states[3]);
    trans_p_dict['M' - 'B'][0] = PyDict_GetItem(trans_p, py_states[1]);
    trans_p_dict['M' - 'B'][1] = PyDict_GetItem(trans_p, py_states[0]);
    trans_p_dict['E' - 'B'][0] = PyDict_GetItem(trans_p, py_states[0]);
    trans_p_dict['E' - 'B'][1] = PyDict_GetItem(trans_p, py_states[1]);
    trans_p_dict['S' - 'B'][0] = PyDict_GetItem(trans_p, py_states[3]);
    trans_p_dict['S' - 'B'][1] = PyDict_GetItem(trans_p, py_states[2]);

    for (i = 0; i < states_num; i++)
    {
        t_dict = PyDict_GetItem(emip_p, py_states[i]);
        t_double = MIN_FLOAT;
        ttemp = PySequence_GetItem(obs, 0);
        if (ttemp)
        {
            item = PyDict_GetItem(t_dict, ttemp);
            Py_DECREF(ttemp);
            if (item)
                t_double = PyFloat_AsDouble(item);
        }
        t_double_2 = PyFloat_AsDouble(PyDict_GetItem(start_p, py_states[i]));
        V[0][states[i] - 'B'] = t_double + t_double_2;
        path[0][states[i] - 'B'] = states[i];
    }

    for (i = 1; i < obs_len; i++)
    {
        t_obs = PySequence_GetItem(obs, i);
        if (!t_obs)
        {
            free(V);
            free(path);
            for (int k = 0; k < states_num; k++)
                Py_DECREF(py_states[k]);
            return NULL; // handle error: failed to get observation item
        }

        for (j = 0; j < states_num; j++)
        {
            em_p = MIN_FLOAT;
            y = states[j];
            item = PyDict_GetItem(emip_p_dict[j], t_obs);
            if (item)
                em_p = PyFloat_AsDouble(item);

            max_prob = MIN_FLOAT;
            best_state = '\0';
            for (p = 0; p < 2; p++)
            {
                prob = em_p;
                y0 = PrevStatus_str[y - 'B'][p];
                prob += V[i - 1][y0 - 'B'];
                item = PyDict_GetItem(trans_p_dict[y - 'B'][p], py_states[j]);
                prob += item ? PyFloat_AsDouble(item) : MIN_FLOAT;

                if (prob > max_prob)
                {
                    max_prob = prob;
                    best_state = y0;
                }
            }

            if (best_state == '\0')
            {
                for (p = 0; p < 2; p++)
                {
                    y0 = PrevStatus_str[y - 'B'][p];
                    if (y0 > best_state)
                        best_state = y0;
                }
            }
            V[i][y - 'B'] = max_prob;
            path[i][y - 'B'] = best_state;
        }
        Py_DECREF(t_obs);
    }

    max_prob = V[obs_len - 1]['E' - 'B'];
    best_state = 'E';

    if (V[obs_len - 1]['S' - 'B'] > max_prob)
    {
        max_prob = V[obs_len - 1]['S' - 'B'];
        best_state = 'S';
    }

    res_tuple = PyTuple_New(2);
    if (!res_tuple)
    {
        free(V);
        free(path);
        for (int k = 0; k < states_num; k++)
            Py_DECREF(py_states[k]);
        return NULL; // handle error: failed to create result tuple
    }

    PyTuple_SetItem(res_tuple, 0, PyFloat_FromDouble(max_prob));
    t_list = PyList_New(obs_len);
    if (!t_list)
    {
        free(V);
        free(path);
        Py_DECREF(res_tuple);
        for (int k = 0; k < states_num; k++)
            Py_DECREF(py_states[k]);
        return NULL; // handle error: failed to create list
    }

    now_state = best_state;
    for (i = obs_len - 1; i >= 0; i--)
    {
        PyList_SetItem(t_list, i, PyUnicode_FromStringAndSize(&now_state, 1));
        now_state = path[i][now_state - 'B'];
    }

    PyTuple_SetItem(res_tuple, 1, t_list);

    free(V);
    free(path);
    for (int k = 0; k < states_num; k++)
        Py_DECREF(py_states[k]);

    return res_tuple;
}

// Python wrapper functions
static PyObject* py_get_dag(PyObject* self, PyObject* args)
{
    PyObject *dag, *freqs, *text;
    if (!PyArg_ParseTuple(args, "OOO", &dag, &freqs, &text))
        return NULL;
    if (get_dag(dag, freqs, text) < 0)
        return NULL; // handle error
    Py_RETURN_NONE;
}

static PyObject* py_get_dag_and_calc(PyObject* self, PyObject* args)
{
    PyObject *freqs, *text, *route;
    double total;
    if (!PyArg_ParseTuple(args, "OOdO", &freqs, &text, &total, &route))
        return NULL;
    if (_get_dag_and_calc(freqs, text, route, total) < 0)
        return NULL; // handle error
    Py_RETURN_NONE;
}

static PyObject* py_viterbi(PyObject* self, PyObject* args)
{
    PyObject *obs, *_states, *start_p, *trans_p, *emip_p;
    if (!PyArg_ParseTuple(args, "OOOOO", &obs, &_states, &start_p, &trans_p, &emip_p))
        return NULL;
    return _viterbi(obs, _states, start_p, trans_p, emip_p);
}

// Method definitions
static PyMethodDef MyMethods[] = {
    {"get_dag", py_get_dag, METH_VARARGS, "Calculate DAG."},
    {"get_dag_and_calc", py_get_dag_and_calc, METH_VARARGS, "Calculate DAG and route."},
    {"viterbi", py_viterbi, METH_VARARGS, "Viterbi algorithm implementation."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef _fast = {
    PyModuleDef_HEAD_INIT,
    "_fast",
    "My module description",
    -1,
    MyMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__fast(void)
{
    return PyModule_Create(&_fast);
}
