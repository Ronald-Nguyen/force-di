public class di_Flow {

    private Flow.Interview flow = null;
    @testVisible
    private Set<String> outputVarsNames;
    @testVisible
    private Set<String> outputVarsNamesRequired;

    public di_Flow(Flow.Interview flow) {
        this.flow = flow;
        init();
    }

    public di_Flow output(String variableName) {
        outputVarsNames.add(variableName);
        return this;
    }

    public di_Flow required(String variableName) {
        output(variableName);
        outputVarsNamesRequired.add(variableName);
        return this;
    }

    public Map<String, Object> run() {
        try {
            flow.start();
            return collectOutputVariables();
        } finally {
            init();
        }
    }

    private Map<String, Object> collectOutputVariables() {
        Map<String, Object> outputVars = new Map<String, Object>();

        for(String outputVar : outputVarsNames) {
            processOutputVariable(outputVars, outputVar);
        }

        return outputVars;
    }

    private void processOutputVariable(Map<String, Object> outputVars, String outputVar) {
        Object outputVarValue = flow.getVariableValue(outputVar);

        if(outputVarValue != null) {
            outputVars.put(outputVar, outputVarValue);
        } else if(outputVarsNamesRequired.contains(outputVar)) {
            throw new FlowException('Output variable ' + outputVar + ' expected but not returned from Flow');
        }
    }

    public Object returning(String outputVariableName) {
        return required(outputVariableName).run().get(outputVariableName);
    }

    private void init() {
        outputVarsNames = new Set<String>();
        outputVarsNamesRequired = new Set<String>();
    }

    public class FlowException extends Exception {}
}