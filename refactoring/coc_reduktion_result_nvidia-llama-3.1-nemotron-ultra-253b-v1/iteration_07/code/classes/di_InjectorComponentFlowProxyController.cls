public with sharing class di_InjectorComponentFlowProxyController {
    public String FlowNameValue { get; set; }
    public Object InputVariablesValue { get; set; }

    public Component.Flow.Interview getInject() {
        Component.Flow.Interview interview = new Component.Flow.Interview();
        interview.name = FlowNameValue;
        if (InputVariablesValue instanceof Map<String, Object>) {
            di_BindingParam.parameters = (Map<String, Object>) InputVariablesValue;
        }
        return interview;
    }
}