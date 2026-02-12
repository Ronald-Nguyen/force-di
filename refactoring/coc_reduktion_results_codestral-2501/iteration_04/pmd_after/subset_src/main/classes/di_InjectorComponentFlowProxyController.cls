public with sharing class di_InjectorComponentFlowProxyController {

    public String FlowNameValue { get; set; }

    public Object InputVariablesValue { get; set; }

    public Component.Flow.Interview getInject() {
        Component.Flow.Interview interview = new Component.Flow.Interview();
        interview.name = FlowNameValue;

        if (InputVariablesValue instanceof Map<Object, Object>) {
            processObjectMap();
        } else if (InputVariablesValue instanceof Map<String, Object>) {
            di_BindingParam.parameters = (Map<String, Object>) InputVariablesValue;
        }

        return interview;
    }

    private void processObjectMap() {
        Map<Object, Object> oldMap = (Map<Object, Object>) InputVariablesValue;
        Map<String, Object> newMap = new Map<String, Object>();

        for (Object key : oldMap.keySet()) {
            newMap.put(String.valueOf(key), oldMap.get(key));
        }

        di_BindingParam.parameters = newMap;
    }
}