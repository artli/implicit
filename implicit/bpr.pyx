import cython
import logging
import multiprocessing
from tqdm.auto import tqdm

from cython.parallel import parallel, prange
from libc.math cimport exp
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport algorithm

import numpy as np

import implicit.cuda

from .recommender_base import MatrixFactorizationBase
from .types cimport floating, integral_1, integral_2, integral_3, numeric


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937) nogil


log = logging.getLogger("implicit")

# thin wrapper around omp_get_thread_num (since referencing directly will cause OSX
# build to fail)
cdef extern from "bpr.h" namespace "implicit" nogil:
    cdef int get_thread_num()


@cython.boundscheck(False)
cdef long long upper_bound(numeric[::1] sorted_list, numeric value) nogil:
    cdef ptrdiff_t index = \
        algorithm.upper_bound(&sorted_list[0], &sorted_list[sorted_list.shape[0]], value) \
        - &sorted_list[0]
    if index >= sorted_list.shape[0]:
        index = -1
    return index


@cython.boundscheck(False)
cdef long long find_index(numeric[::1] sorted_list, numeric value) nogil:
    cdef long long index = upper_bound(sorted_list, value) - 1
    if index < 0 or sorted_list[index] != value:
        index = -1
    return index


@cython.boundscheck(False)
cdef int find_row_number(integral_1[::1] indptr, integral_2 cell_index) nogil:
    cdef long long row_number = upper_bound(indptr, cell_index) - 1
    if row_number < 0:
        row_number = -1
    return row_number


@cython.boundscheck(False)
cdef long long find_entry_index(integral_1[::1] indices, integral_2[::1] indptr,
                                integral_3 row, integral_3 col) nogil:
    cdef long long index_in_row = find_index(indices[indptr[row] : indptr[row + 1]], col)
    if index_in_row == -1:
        return -1
    return indptr[row] + index_in_row


@cython.boundscheck(False)
cdef bool is_liked(integral_1[::1] indices, integral_2[::1] indptr, numeric[:] ratings,
                   integral_3 row, integral_3 col) nogil:
    """ Given a CSR matrix, returns whether the [rowid, colid] contains a non zero.
    Assumes the CSR matrix has sorted indices """
    cdef long long index = find_entry_index(indices, indptr, row, col)
    return index != -1 and ratings[index] == 1


cdef class RNGVector(object):
    """ This class creates one c++ rng object per thread, and enables us to randomly sample
    liked/disliked items here in a thread safe manner """
    cdef vector[mt19937] rng
    cdef vector[uniform_int_distribution[long]]  dist

    def __init__(self, int num_threads, long rows):
        for i in range(num_threads):
            self.rng.push_back(mt19937(np.random.randint(2**31)))
            self.dist.push_back(uniform_int_distribution[long](0, rows))

    cdef inline long generate(self, int thread_id) nogil:
        return self.dist[thread_id](self.rng[thread_id])


class BayesianPersonalizedRanking(MatrixFactorizationBase):
    """ Bayesian Personalized Ranking

    A recommender model that learns  a matrix factorization embedding based off minimizing the
    pairwise ranking loss described in the paper `BPR: Bayesian Personalized Ranking from Implicit
    Feedback <https://arxiv.org/pdf/1205.2618.pdf>`_.

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for SGD updates during training
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_gpu : bool, optional
        Fit on the GPU if available
    iterations : int, optional
        The number of training epochs to use when fitting the data
    verify_negative_samples: bool, optional
        When sampling negative items, check if the randomly picked negative item has actually
        been liked by the user. This check increases the time needed to train but usually leads
        to better predictions.
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """
    def __init__(self, factors=100, learning_rate=0.01, regularization=0.01, dtype=np.float32,
                 iterations=100, implicit_prob=1.0, weighted_negatives=True, item_biases=True,
                 use_gpu=implicit.cuda.HAS_CUDA, num_threads=0,
                 verify_negative_samples=True):
        super(BayesianPersonalizedRanking, self).__init__()

        self.non_bias_factors = factors
        self.bias_factors = int(item_biases)
        self.total_factors = self.bias_factors + self.non_bias_factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.dtype = dtype
        self.implicit_prob = implicit_prob
        self.weighted_negatives = weighted_negatives
        self.item_biases = item_biases
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.verify_negative_samples = verify_negative_samples

        if use_gpu and self.total_factors % 32:
            padding = 32 - self.total_factors % 32
            log.warning(
                "GPU training requires total factor count to be a multiple of 32."
                " Increasing the number of non-bias factors from %i to %i.",
                self.non_bias_factors, self.non_bias_factors + padding)
            self.non_bias_factors += padding
        self.mean_rating = None

    def _random_factor_matrix(self, row_count):
        result = np.random.rand(row_count, self.total_factors).astype(self.dtype) - .5
        result /= self.non_bias_factors
        return result

    def random_item_matrix(self, row_count):
        return self._random_factor_matrix(row_count)

    def random_user_matrix(self, row_count):
        result = self._random_factor_matrix(row_count)
        if self.item_biases:
            result[:, -1] = 1
        return result

    def _exact_num_threads(self):
        # we accept num_threads = 0 as indicating to create as many threads as we have cores,
        # but in that case we need the number of cores, since we need to initialize RNG state per
        # thread. Get the appropiate value back from openmp
        if self.num_threads == 0:
            return multiprocessing.cpu_count()
        return self.num_threads

    def _train(
            self, user_items, learning_rate, regularization, implicit_prob, one_step_per_user,
            user_factors, item_factors, update_items, iterations, show_progress=True):
        cdef int num_threads = self._exact_num_threads()
        cdef RNGVector rng = RNGVector(num_threads, len(user_items.data) - 1)
        cdef int steps_per_iteration = user_items.shape[0] if one_step_per_user else len(user_items.data)
        with tqdm(total=iterations, disable=not show_progress) as progress:
            for epoch in range(iterations):
                correct, skipped = bpr_epoch(
                    rng, user_items.data, user_items.indices, user_items.indptr,
                    user_factors, item_factors,
                    learning_rate, regularization,
                    implicit_prob, one_step_per_user, self.weighted_negatives,
                    self.item_biases, update_items,
                    num_threads, self.verify_negative_samples)
                progress.update(1)
                def to_percents(value, total):
                    if total == 0:
                        return 0
                    return 100.0 * value / total
                progress.set_postfix({
                    "correct": "%.2f%%" % to_percents(correct, steps_per_iteration - skipped),
                    "skipped": "%.2f%%" % to_percents(skipped, steps_per_iteration)})

    def compute_user_profiles(
            self, user_items, iterations, learning_rate=None, regularization=None,
            implicit_prob=None, initial_profiles=None, show_progress=True):
        profiles = initial_profiles
        if profiles is None:
            profiles = self._random_factor_matrix(row_count=user_items.shape[0])
        if learning_rate is None:
            learning_rate = self.learning_rate
        if regularization is None:
            regularization = self.regularization
        if implicit_prob is None:
            implicit_prob = self.implicit_prob
        self._train(
            user_items, learning_rate, regularization, implicit_prob, True,
            profiles, self.item_factors, False, iterations, show_progress)
        return profiles

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def fit(self, item_users=None, user_items=None, show_progress=True):
        """ Factorizes the item_users matrix

        Parameters
        ----------
        item_users: coo_matrix
            Matrix of confidences for the liked items. This matrix should be a coo_matrix where
            the rows of the matrix are the item, and the columns are the users that liked that item.
            BPR ignores the weight value of the matrix right now - it treats non zero entries
            as a binary signal that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar
        """
        assert (item_users is None) != (user_items is None)
        if user_items is None:
            user_items = item_users.T.tocsr()
        if not user_items.has_sorted_indices:
            user_items.sort_indices()

        self.mean_rating = user_items.data.mean()

        users, items = user_items.shape

        # create factors if not already created.
        # Note: the final dimension is for the item bias term - which is set to a 1 for all users
        # this simplifies interfacing with approximate nearest neighbours libraries etc
        if self.item_factors is None:
            self.item_factors = self.random_item_matrix(items)
            # set factors to all zeros for items without any ratings
            item_counts = np.bincount(user_items.indices, minlength=items)
            self.item_factors[item_counts == 0] = np.zeros(self.total_factors)

        if self.user_factors is None:
            self.user_factors = self.random_user_matrix(users)
            # set factors to all zeros for users without any ratings
            user_counts = np.ediff1d(user_items.indptr)
            self.user_factors[user_counts == 0] = 0

        if self.use_gpu:
            raise NotImplementedError
            # return self._fit_gpu(user_items, userids, show_progress)

        self._train(
            user_items, self.learning_rate, self.regularization, self.implicit_prob, False,
            self.user_factors, self.item_factors, True, self.iterations, show_progress)

    def _fit_gpu(self, user_items, userids_host, show_progress=True):
        if not implicit.cuda.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        if self.dtype == np.float64:
            log.warning("Factors of dtype float64 aren't supported with gpu fitting. "
                        "Converting factors to float32")
            self.user_factors = self.user_factors.astype(np.float32)
            self.item_factors = self.item_factors.astype(np.float32)

        userids = implicit.cuda.CuIntVector(userids_host)
        itemids = implicit.cuda.CuIntVector(user_items.indices)
        indptr = implicit.cuda.CuIntVector(user_items.indptr)

        X = implicit.cuda.CuDenseMatrix(self.user_factors)
        Y = implicit.cuda.CuDenseMatrix(self.item_factors)

        log.debug("Running %i BPR training epochs", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                correct, skipped = implicit.cuda.cu_bpr_update(userids, itemids, indptr,
                                                               X, Y, self.learning_rate,
                                                               self.regularization,
                                                               np.random.randint(2**31),
                                                               self.verify_negative_samples)
                progress.update(1)
                total = len(user_items.data)
                progress.set_postfix({"correct": "%.2f%%" % (100.0 * correct / (total - skipped)),
                                      "skipped": "%.2f%%" % (100.0 * skipped / total)})

        X.to_host(self.user_factors)
        Y.to_host(self.item_factors)


@cython.cdivision(True)
@cython.boundscheck(True)
cdef bool bpr_update_profiles(
        floating *user, floating *liked, floating *disliked, int total_factors,
        float learning_rate, float reg, bool item_biases, bool update_items) nogil:
    cdef int non_bias_factors = total_factors - <int>item_biases
    cdef int j
    cdef floating z, score, temp

    # compute the score
    score = 0
    for j in range(total_factors):
        score = score + user[j] * (liked[j] - disliked[j])
    z = 1.0 / (1.0 + exp(score))

    # update the factors via sgd.
    for j in range(non_bias_factors):
        temp = user[j]
        user[j] += learning_rate * (z * (liked[j] - disliked[j]) - reg * user[j])
        if update_items:
            liked[j] += learning_rate * (z * temp - reg * liked[j])
            disliked[j] += learning_rate * (-z * temp - reg * disliked[j])

    if item_biases and update_items:
        # update item bias terms (last column of factorized matrix)
        liked[non_bias_factors] += learning_rate * (z - reg * liked[non_bias_factors])
        disliked[non_bias_factors] += learning_rate * (-z - reg * disliked[non_bias_factors])

    return z < .5


@cython.cdivision(True)
@cython.boundscheck(False)
cdef long long sample_user_interaction_index(
        RNGVector rng, int thread_id, integral_1[::1] itemids, integral_2[::1] indptr, int user_id) nogil:
    cdef long long user_interaction_count, interaction_number
    if user_id == -1:
        return -1
    user_interaction_count = indptr[user_id + 1] - indptr[user_id]
    if user_interaction_count == -1:
        return -1
    interaction_number = rng.generate(thread_id) % user_interaction_count
    return indptr[user_id] + interaction_number


@cython.cdivision(True)
@cython.boundscheck(True)
def bpr_epoch(
        RNGVector rng,
        numeric[:] ratings, integral_1[::1] itemids, integral_2[::1] indptr,
        floating[:, :] X, floating[:, :] Y,
        float learning_rate, float reg,
        float implicit_prob, bool one_step_per_user, bool weighted_negatives,
        bool item_biases, bool update_items,
        int num_threads, bool verify_neg):
    if one_step_per_user and update_items:
        raise NotImplementedError(
            'Proper user shuffling is not implemented in the per-user sampling scheme')

    cdef int users = X.shape[0]
    cdef int items = Y.shape[0]
    cdef int total_factors = Y.shape[1]
    cdef long long sample_count, i, liked_index, disliked_index
    cdef long long correct = 0
    cdef long long skipped1 = 0
    cdef long long skipped2 = 0
    cdef int user_id = -1
    cdef int liked_id = -1
    cdef int disliked_id = -1
    cdef int thread_id
    cdef int j
    cdef bool unknown, wrong_sample
    cdef float temp

    cdef floating * user
    cdef floating * liked
    cdef floating * disliked

    if one_step_per_user:
        sample_count = users
    else:
        sample_count = len(ratings)

    with nogil, parallel(num_threads=num_threads):
        thread_id = get_thread_num()

        for i in prange(sample_count, schedule='guided'):
            if one_step_per_user:
                user_id = i
                if indptr[user_id + 1] - indptr[user_id] == 0:
                    continue

            for j in range(20):
                if one_step_per_user:
                    liked_index = sample_user_interaction_index(rng, thread_id, itemids, indptr, user_id)
                else:
                    liked_index = rng.generate(thread_id)
                if liked_index != -1 and ratings[liked_index] == 1:
                    break
            else:
                skipped1 += 1
                continue

            liked_id = itemids[liked_index]
            if not one_step_per_user:
                user_id = find_row_number(indptr, liked_index)

            temp = rng.generate(thread_id) / <float>ratings.shape[0]
            if temp < implicit_prob:
                if weighted_negatives:
                    disliked_index = rng.generate(thread_id)
                    disliked_id = itemids[disliked_index]
                else:
                    disliked_id = rng.generate(thread_id) % items
            else:
                disliked_index = sample_user_interaction_index(rng, thread_id, itemids, indptr, user_id)
                disliked_id = itemids[disliked_index]

            unknown = user_id == -1 or liked_id == -1 or disliked_id == -1
            # if the user has liked the item, skip this for now
            wrong_sample = verify_neg and is_liked(itemids, indptr, ratings, user_id, disliked_id)
            if unknown or wrong_sample:
                skipped2 += 1
                continue

            # get pointers to the relevant factors
            user, liked, disliked = &X[user_id, 0], &Y[liked_id, 0], &Y[disliked_id, 0]

            if bpr_update_profiles(
                    user, liked, disliked, total_factors,
                    learning_rate, reg, item_biases, update_items):
                correct += 1

    return correct, skipped1 + skipped2
