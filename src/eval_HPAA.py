from .detectors import *

def eval_HPAA(args):
    eval_hpaa = eval_HPAA_samples(args)
    eval_hpaa.eval()


class eval_HPAA_samples():
    def __init__(self, args):
        self.file_eval = args.file_eval
        self.hpaa_folder = args.hpaa_folder
        
        x_eval = []
        for idx, filename in enumerate(self.file_eval):
            df = pd.read_csv(filename)
            x_eval.extend(df.x0.tolist())
        
        self.x_eval = x_eval
        eval_prefix = getattr(args, "eval_prefix", "eval")
        self.eval_prefix = eval_prefix
        self.model_name = args.detector_name
        
        print(f"\n>>> Loading config of {self.model_name} <<<\n")
        with open("./src/detectors.yaml", 'r') as file:
            self.model_configs = yaml.safe_load(file)
        self.download_path = self.model_configs["download_path"]
        self.call_type = self.model_configs["models"][self.model_name]["call_type"]
        
        # model init
        self.classification_models = ["shieldgemma-2b", "shieldgemma-9b"]
        self.model_init()
        
        params = (
            "do_sample", "temperature", "top_p", "top_k",
            "tau", "bias_yes", "bias_no", "min_margin"
        )
        self.kwargs = self._filter_detect_kwargs({k: getattr(args, k) for k in params})
        
        # prompt init
        if self.model_configs["models"][self.model_name]["text_only"]:
            self.prompt = None
        else:
            self.prompt = self.model_configs["models"][self.model_name]["prompt_presets"]
    
    @classmethod
    def from_args(cls, args, prompt_val):
        return cls(model_name=args.model_name)

    def model_init(self):
        call_map = {
            "pipeline": lambda: model_pipeline(self.model_name, self.download_path),
            "regular": lambda: model_regular(self.model_name, self.download_path, self.classification_models),
            "api": lambda: model_api(self.model_name)
        }
        func = call_map.get(self.call_type)
        self.detector = func()

    @timeit
    def eval(self):
        if self.prompt is None:
            inputs = self.x_eval
        else:
            inputs = [self.prompt.replace("<Adversarial Text>", t) for t in self.x_eval]
        
        outputs = [self.detector.detect(t, **self.kwargs) for t in inputs]

        detector_config = dict(self.kwargs) if self.kwargs is not None else {}
        
        detect = {
            "x": inputs,
            "eval": outputs,
            **detector_config
        }
        df = pd.DataFrame(detect)

        current_time = datetime.now()
        current_time_formatted = current_time.strftime("%Y%m%d%H%M%S")
        
        outpath = os.path.join(self.hpaa_folder, f"{self.eval_prefix}.{self.model_name}.{current_time_formatted}.csv")
        df.to_csv(outpath, index=False)
    
    def _filter_detect_kwargs(self, overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not overrides:
            return {}
        sig = inspect.signature(self.detector.detect)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values())
        if accepts_var_kw:
            return {k: v for k, v in overrides.items() if v is not None}
        return {k: v for k, v in overrides.items() if (v is not None and k in params)}