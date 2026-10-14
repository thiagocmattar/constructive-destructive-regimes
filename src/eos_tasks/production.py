"""Production-only parity with the existing MLP runner and solver retries."""
from scipy.sparse.linalg import ArpackNoConvergence

from .task import Task
from src.cifar_eos.core import full_gradient


class ProductionTask(Task):
    def gradient(self, ids=None):
        if self.name=="mlp" and ids is None:
            loss,_,gradient=full_gradient(self.model,self.data["x_train"],self.data["y_train"],self.physical,"mse")
            return loss,gradient.clone()
        return super().gradient(ids)

    def sharpness(self, ids, options):
        attempts=[]
        settings=[options]
        if options.get("fallback"):
            settings.append(dict(options,**options["fallback"]))
        for setting in settings:
            try:
                result=super().sharpness(ids,setting)
                attempts.append({"status":"passed","ncv":setting["ncv"],"maxiter":setting["maxiter"]})
                result["solver_attempts"]=attempts
                return result
            except (ArpackNoConvergence,RuntimeError) as error:
                if isinstance(error,RuntimeError) and "Eigenpair residual failed" not in str(error):
                    raise
                attempts.append({"status":"failed","ncv":setting["ncv"],"maxiter":setting["maxiter"],"error":str(error)})
        raise RuntimeError(f"All prespecified eigenpair solves failed: {attempts}")
