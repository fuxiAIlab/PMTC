import collections
import numpy as np


class CurriculumLearner:
    def __init__(self, init_alpha, max_alpha, update_step_value):
        self._alpha = init_alpha
        self._max_alpha = max_alpha
        self._update_coeff = 1.005
        self._update_step_value = update_step_value
        self._current_step = 0
        self._trained_task = []

    @property
    def trained_task(self):
        return self._trained_task

    @property
    def current_step(self):
        return self._current_step

    @property
    def update_step_value(self):
        return self._update_step_value

    @property
    def max_alpha(self):
        return self._max_alpha

    @property
    def alpha(self):
        return min(self._alpha, self.max_alpha)

    def reset_train_task(self):
        self._trained_task = []

    def add_train_task(self, task_name):
        self._trained_task.append(task_name)


class LinearTaskCurriculumLearner(CurriculumLearner):
    def __init__(self, alpha, max_alpha, update_step_value):
        super().__init__(alpha, max_alpha, update_step_value)

    def update_step(self):
        self._current_step += 1

    def adjust_alpha(self):
        if self.current_step % self.update_step_value == 0:
            if self._alpha < self.max_alpha:
                self._alpha *= self._update_coeff
                print(f"[Curriculum Learner] Adjust alpha to {self.alpha}, max_alpha: {self.max_alpha}")


class BpeAdjustMaskCurriculumLearner():
    def __init__(self,
                 update_step_value,
                 init_mlm_prob,
                 collate_fn,
                 bpe_cr_update_coeff=1.01,
                 bpe_cr_adjust_tol=1e-4,
                 is_anti_curr=False
                 ):
        self.max_mask_ratio = 0.5
        self.adjust_tol = bpe_cr_adjust_tol
        self.update_step_value = update_step_value
        self._update_coeff = bpe_cr_update_coeff
        self._current_step = 0
        self.mlm_prob = init_mlm_prob
        self.losses = []
        self.collate_fn = collate_fn

        if is_anti_curr:
            print("Trigger Anti Curr!!!!!!!!!")
            self.min_mask_ratio = init_mlm_prob
            self.mlm_prob = self.max_mask_ratio
            setattr(self.collate_fn, f'mlm_probability', self.mlm_prob)

        self.is_anti_curr = is_anti_curr

    @property
    def current_step(self):
        return self._current_step

    def update_step(self):
        self._current_step += 1

    def _reset_task_loss(self):
        self.losses = []

    def update_loss(self, loss_value):
        self.losses.append(loss_value)

    def adjust_mlm_prob(self, verbose=0):
        if self.current_step % self.update_step_value == 0:
            overall_decrease = (self.losses[0] - self.losses[-1]) / len(self.losses)
            avg_step_loss = np.average([x1 - x2 for x1, x2 in zip(self.losses[:-1], self.losses[1:])])
            if overall_decrease > 0 and avg_step_loss > self.adjust_tol:

                if self.is_anti_curr:
                    self.mlm_prob = max(self.min_mask_ratio, self.mlm_prob / self._update_coeff)
                else:
                    self.mlm_prob = min(self.max_mask_ratio, self.mlm_prob * self._update_coeff)
                setattr(self.collate_fn, f'mlm_probability', self.mlm_prob)
                if verbose >= 2:
                    print(
                        f"Adjust BPE mlm prob, step for previous mlm prob: {len(self.losses)}, "
                        f"overall_decrease: {overall_decrease},"
                        f" avg_step_loss: {avg_step_loss}, "
                        f" adjust_tol: {self.adjust_tol}"
                        f"set mask ratio to {self.mlm_prob}")
            else:
                if verbose >= 2:
                    print(f"BPE mlm prob not changed,"
                          f" step for previous mlm prob: {len(self.losses)},"
                          f" overall_decrease: {overall_decrease},"
                          f" avg_step_loss: {avg_step_loss},"
                          f" adjust_tol: {self.adjust_tol}"
                          f".")
            self._reset_task_loss()


class AdjustMaskCurriculumLearner():
    def __init__(self, update_step_value):
        self.max_mask_ratio = 0.9
        self.adjust_tol = 1e-4
        self.update_step_value = update_step_value
        self._update_coeff = 1.005
        self._current_step = 0
        self.task_mlm_prob = {}
        self.task_losses = collections.defaultdict(lambda: [])

    @property
    def current_step(self):
        return self._current_step

    def add_collate_cn(self, collate_fn):
        self.collate_fn = collate_fn
        if collate_fn.is_task0:
            self.task_mlm_prob['task0'] = collate_fn.task0_mlm_prob
        if collate_fn.is_task1:
            self.task_mlm_prob['task1'] = collate_fn.task1_mlm_prob
        if collate_fn.is_task2:
            self.task_mlm_prob['task2'] = collate_fn.task2_mlm_prob
        if collate_fn.is_task3:
            self.task_mlm_prob['task3'] = collate_fn.task3_mlm_prob
        if collate_fn.is_task4:
            self.task_mlm_prob['task4'] = collate_fn.task4_mlm_prob

        # assertion
        for task_name, mlm_prob in self.task_mlm_prob.items():
            assert mlm_prob > 0

    def update_step(self):
        self._current_step += 1

    def _reset_task_loss(self):
        self.task_losses = collections.defaultdict(lambda: [])

    def add_task_loss(self, task_name, loss):
        self.task_losses[task_name].append(loss)

    def adjust_mlm_prob(self, verbose=0):
        if self.current_step % self.update_step_value == 0:
            for task_name, losses in self.task_losses.items():
                overall_decrease = (losses[0] - losses[-1]) / len(losses)
                avg_step_loss = np.average([x1 - x2 for x1, x2 in zip(losses[:-1], losses[1:])])
                if overall_decrease > 0 and avg_step_loss > self.adjust_tol:
                    self.task_mlm_prob[task_name] = min(self.max_mask_ratio,
                                                        self.task_mlm_prob[task_name] * self._update_coeff)
                    setattr(self.collate_fn, f'{task_name}_mlm_prob', self.task_mlm_prob[task_name])
                    if verbose >= 2:
                        print(
                            f"Adjust task-{task_name}, step: {len(losses)}, overall_decrease: {overall_decrease},"
                            f" avg_step_loss: {avg_step_loss}, "
                            f" adjust_tol: {self.adjust_tol}"
                            f"set mask ratio to {self.task_mlm_prob[task_name]}")
                else:
                    if verbose >= 2:
                        print(f"Task-{task_name} mlm prob not changed,"
                              f" step: {len(losses)},"
                              f" overall_decrease: {overall_decrease},"
                              f" avg_step_loss: {avg_step_loss},"
                              f" adjust_tol: {self.adjust_tol}"
                              f".")
            self._reset_task_loss()
