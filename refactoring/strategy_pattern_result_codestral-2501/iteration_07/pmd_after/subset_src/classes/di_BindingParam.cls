public with sharing class di_BindingParam {

    public static Map<String, Object> parameters { get; set; }

    @InvocableMethod(
        label = 'Get Value'
    )
    public static List<Response> invoke( List<Request> requests ) {

        List<Response> responses = new List<Response>();

        for ( Request req : requests ) {

            Response res = new Response();

            if ( parameters != null ) {

                Object value = parameters.get( req.paramName );

                if ( value != null ) {

                    if ( value instanceof Decimal ) {

                        Decimal decimalValue = (Decimal) value;
                        res.longValue = toLongValue( decimalValue );
                        res.decimalValue = decimalValue;
                        res.stringValue = toStringValue( decimalValue );
                        res.booleanValue = toBooleanValue( decimalValue );
                        res.dateValue = toDateValue( decimalValue );
                        res.dateTimeValue = toDateTimeValue( decimalValue );

                    } else if ( value instanceof Decimal[] ) {

                        Decimal[] decimalValues = (Decimal[]) value;
                        res.decimalValues = decimalValues;
                        for ( Decimal decimalValue : decimalValues ) {
                            res.longValues.add( toLongValue( decimalValue ) );
                            res.stringValues.add( toStringValue( decimalValue ) );
                            res.booleanValues.add( toBooleanValue( decimalValue ) );
                            res.dateValues.add( toDateValue( decimalValue ) );
                            res.dateTimeValues.add( toDateTimeValue( decimalValue ) );
                        }

                    } else if ( value instanceof String ) {

                        String stringValue = (String) value;
                        res.longValue = toLongValue( stringValue );
                        res.decimalValue = toDecimalValue( stringValue );
                        res.stringValue = stringValue;
                        res.booleanValue = toBooleanValue( stringValue );
                        res.dateValue = toDateValue( stringValue );
                        res.dateTimeValue = toDateTimeValue( stringValue );

                    } else if ( value instanceof String[] ) {

                        String[] stringValues = (String[]) value;
                        res.stringValues = stringValues;
                        for ( String stringValue : stringValues ) {
                            res.longValues.add( toLongValue( stringValue ) );
                            res.decimalValues.add( toDecimalValue( stringValue ) );
                            res.booleanValues.add( toBooleanValue( stringValue ) );
                            res.dateValues.add( toDateValue( stringValue ) );
                            res.dateTimeValues.add( toDateTimeValue( stringValue ) );
                        }

                    } else if ( value instanceof Boolean ) {

                        Boolean booleanValue = (Boolean) value;
                        res.longValue = toLongValue( booleanValue );
                        res.decimalValue = toDecimalValue( booleanValue );
                        res.stringValue = toStringValue( booleanValue );
                        res.booleanValue = booleanValue;
                        res.dateValue = toDateValue( booleanValue );
                        res.dateTimeValue = toDateTimeValue( booleanValue );

                    } else if ( value instanceof Boolean[] ) {

                        Boolean[] booleanValues = (Boolean[]) value;
                        res.booleanValues = booleanValues;
                        for ( Boolean booleanValue : booleanValues ) {
                            res.longValues.add( toLongValue( booleanValue ) );
                            res.decimalValues.add( toDecimalValue( booleanValue ) );
                            res.stringValues.add( toStringValue( booleanValue ) );
                            res.dateValues.add( toDateValue( booleanValue ) );
                            res.dateTimeValues.add( toDateTimeValue( booleanValue ) );
                        }

                    } else if ( value instanceof Date ) {

                        Date dateValue = (Date) value;
                        res.longValue = toLongValue( dateValue );
                        res.decimalValue = toDecimalValue( dateValue );
                        res.stringValue = toStringValue( dateValue );
                        res.booleanValue = toBooleanValue( dateValue );
                        res.dateValue = dateValue;
                        res.dateTimeValue = toDateTimeValue( dateValue );

                    } else if ( value instanceof Date[] ) {

                        Date[] dateValues = (Date[]) value;
                        res.dateValues = dateValues;
                        for ( Date dateValue : dateValues ) {
                            res.longValues.add( toLongValue( dateValue ) );
                            res.decimalValues.add( toDecimalValue( dateValue ) );
                            res.stringValues.add( toStringValue( dateValue ) );
                            res.booleanValues.add( toBooleanValue( dateValue ) );
                            res.dateTimeValues.add( toDateTimeValue( dateValue ) );
                        }

                    } else if ( value instanceof DateTime ) {

                        DateTime dateTimeValue = (DateTime) value;
                        res.longValue = toLongValue( dateTimeValue );
                        res.decimalValue = toDecimalValue( dateTimeValue );
                        res.stringValue = toStringValue( dateTimeValue );
                        res.booleanValue = toBooleanValue( dateTimeValue );
                        res.dateValue = toDateValue( dateTimeValue );
                        res.dateTimeValue = dateTimeValue;

                    } else if ( value instanceof DateTime[] ) {

                        DateTime[] dateTimeValues = (DateTime[]) value;
                        res.dateTimeValues = dateTimeValues;
                        for ( DateTime dateTimeValue : dateTimeValues ) {
                            res.longValues.add( toLongValue( dateTimeValue ) );
                            res.decimalValues.add( toDecimalValue( dateTimeValue ) );
                            res.stringValues.add( toStringValue( dateTimeValue ) );
                            res.booleanValues.add( toBooleanValue( dateTimeValue ) );
                            res.dateValues.add( toDateValue( dateTimeValue ) );
                        }

                    }

                    if ( res.longValue     == null && !res.longValues.isEmpty() )     { res.longValue = res.longValues[0]; }
                    if ( res.decimalValue  == null && !res.decimalValues.isEmpty() )  { res.decimalValue = res.decimalValues[0]; }
                    if ( res.stringValue   == null && !res.stringValues.isEmpty() )   { res.stringValue = res.stringValues[0]; }
                    if ( res.booleanValue  == null && !res.booleanValues.isEmpty() )  { res.booleanValue = res.booleanValues[0]; }
                    if ( res.dateValue     == null && !res.dateValues.isEmpty() )     { res.dateValue = res.dateValues[0]; }
                    if ( res.dateTimeValue == null && !res.dateTimeValues.isEmpty() ) { res.dateTimeValue = res.dateTimeValues[0]; }

                    if ( res.longValue     != null && res.longValues.isEmpty() )     { res.longValues.add( res.longValue ); }
                    if ( res.decimalValue  != null && res.decimalValues.isEmpty() )  { res.decimalValues.add( res.decimalValue ); }
                    if ( res.stringValue   != null && res.stringValues.isEmpty() )   { res.stringValues.add( res.stringValue ); }
                    if ( res.booleanValue  != null && res.booleanValues.isEmpty() )  { res.booleanValues.add( res.booleanValue ); }
                    if ( res.dateValue     != null && res.dateValues.isEmpty() )     { res.dateValues.add( res.dateValue ); }
                    if ( res.dateTimeValue != null && res.dateTimeValues.isEmpty() ) { res.dateTimeValues.add( res.dateTimeValue ); }

                } else {

                    System.debug( LoggingLevel.WARN, 'BindingParam.parameters.get("' + req + '") is null' );

                }

            } else {

                System.debug( LoggingLevel.WARN, 'BindingParam.parameters is null' );

            }

            System.debug( 'paramName=' + req );
            System.debug( res );

            responses.add( res );

        }

        return responses;
    }

    private interface ConversionStrategy {
        Response execute(Object value, Response response);
    }

    private class DecimalConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            Decimal decimalValue = (Decimal) value;
            response.longValue = toLongValue(decimalValue);
            response.decimalValue = decimalValue;
            response.stringValue = toStringValue(decimalValue);
            response.booleanValue = toBooleanValue(decimalValue);
            response.dateValue = toDateValue(decimalValue);
            response.dateTimeValue = toDateTimeValue(decimalValue);
            return response;
        }
    }

    private class DecimalArrayConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            Decimal[] decimalValues = (Decimal[]) value;
            response.decimalValues = decimalValues;
            for (Decimal decimalValue : decimalValues) {
                response.longValues.add(toLongValue(decimalValue));
                response.stringValues.add(toStringValue(decimalValue));
                response.booleanValues.add(toBooleanValue(decimalValue));
                response.dateValues.add(toDateValue(decimalValue));
                response.dateTimeValues.add(toDateTimeValue(decimalValue));
            }
            return response;
        }
    }

    private class StringConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            String stringValue = (String) value;
            response.longValue = toLongValue(stringValue);
            response.decimalValue = toDecimalValue(stringValue);
            response.stringValue = stringValue;
            response.booleanValue = toBooleanValue(stringValue);
            response.dateValue = toDateValue(stringValue);
            response.dateTimeValue = toDateTimeValue(stringValue);
            return response;
        }
    }

    private class StringArrayConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            String[] stringValues = (String[]) value;
            response.stringValues = stringValues;
            for (String stringValue : stringValues) {
                response.longValues.add(toLongValue(stringValue));
                response.decimalValues.add(toDecimalValue(stringValue));
                response.booleanValues.add(toBooleanValue(stringValue));
                response.dateValues.add(toDateValue(stringValue));
                response.dateTimeValues.add(toDateTimeValue(stringValue));
            }
            return response;
        }
    }

    private class BooleanConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            Boolean booleanValue = (Boolean) value;
            response.longValue = toLongValue(booleanValue);
            response.decimalValue = toDecimalValue(booleanValue);
            response.stringValue = toStringValue(booleanValue);
            response.booleanValue = booleanValue;
            response.dateValue = toDateValue(booleanValue);
            response.dateTimeValue = toDateTimeValue(booleanValue);
            return response;
        }
    }

    private class BooleanArrayConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            Boolean[] booleanValues = (Boolean[]) value;
            response.booleanValues = booleanValues;
            for (Boolean booleanValue : booleanValues) {
                response.longValues.add(toLongValue(booleanValue));
                response.decimalValues.add(toDecimalValue(booleanValue));
                response.stringValues.add(toStringValue(booleanValue));
                response.dateValues.add(toDateValue(booleanValue));
                response.dateTimeValues.add(toDateTimeValue(booleanValue));
            }
            return response;
        }
    }

    private class DateConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            Date dateValue = (Date) value;
            response.longValue = toLongValue(dateValue);
            response.decimalValue = toDecimalValue(dateValue);
            response.stringValue = toStringValue(dateValue);
            response.booleanValue = toBooleanValue(dateValue);
            response.dateValue = dateValue;
            response.dateTimeValue = toDateTimeValue(dateValue);
            return response;
        }
    }

    private class DateArrayConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            Date[] dateValues = (Date[]) value;
            response.dateValues = dateValues;
            for (Date dateValue : dateValues) {
                response.longValues.add(toLongValue(dateValue));
                response.decimalValues.add(toDecimalValue(dateValue));
                response.stringValues.add(toStringValue(dateValue));
                response.booleanValues.add(toBooleanValue(dateValue));
                response.dateTimeValues.add(toDateTimeValue(dateValue));
            }
            return response;
        }
    }

    private class DateTimeConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            DateTime dateTimeValue = (DateTime) value;
            response.longValue = toLongValue(dateTimeValue);
            response.decimalValue = toDecimalValue(dateTimeValue);
            response.stringValue = toStringValue(dateTimeValue);
            response.booleanValue = toBooleanValue(dateTimeValue);
            response.dateValue = toDateValue(dateTimeValue);
            response.dateTimeValue = dateTimeValue;
            return response;
        }
    }

    private class DateTimeArrayConversionStrategy implements ConversionStrategy {
        public Response execute(Object value, Response response) {
            DateTime[] dateTimeValues = (DateTime[]) value;
            response.dateTimeValues = dateTimeValues;
            for (DateTime dateTimeValue : dateTimeValues) {
                response.longValues.add(toLongValue(dateTimeValue));
                response.decimalValues.add(toDecimalValue(dateTimeValue));
                response.stringValues.add(toStringValue(dateTimeValue));
                response.booleanValues.add(toBooleanValue(dateTimeValue));
                response.dateValues.add(toDateValue(dateTimeValue));
            }
            return response;
        }
    }

    private static Map<Type, ConversionStrategy> STRATEGIES = new Map<Type, ConversionStrategy>{
        Decimal.class => new DecimalConversionStrategy(),
        Decimal[].class => new DecimalArrayConversionStrategy(),
        String.class => new StringConversionStrategy(),
        String[].class => new StringArrayConversionStrategy(),
        Boolean.class => new BooleanConversionStrategy(),
        Boolean[].class => new BooleanArrayConversionStrategy(),
        Date.class => new DateConversionStrategy(),
        Date[].class => new DateArrayConversionStrategy(),
        DateTime.class => new DateTimeConversionStrategy(),
        DateTime[].class => new DateTimeArrayConversionStrategy()
    };

    private static Response processValue(Object value, Response response) {
        if (value != null) {
            Type valueType = value.getClass();
            if (STRATEGIES.containsKey(valueType)) {
                response = STRATEGIES.get(valueType).execute(value, response);
            }
        }
        return response;
    }

    private static Response setDefaultValues(Response response) {
        if (response.longValue == null && !response.longValues.isEmpty()) {
            response.longValue = response.longValues[0];
        }
        if (response.decimalValue == null && !response.decimalValues.isEmpty()) {
            response.decimalValue = response.decimalValues[0];
        }
        if (response.stringValue == null && !response.stringValues.isEmpty()) {
            response.stringValue = response.stringValues[0];
        }
        if (response.booleanValue == null && !response.booleanValues.isEmpty()) {
            response.booleanValue = response.booleanValues[0];
        }
        if (response.dateValue == null && !response.dateValues.isEmpty()) {
            response.dateValue = response.dateValues[0];
        }
        if (response.dateTimeValue == null && !response.dateTimeValues.isEmpty()) {
            response.dateTimeValue = response.dateTimeValues[0];
        }

        if (response.longValue != null && response.longValues.isEmpty()) {
            response.longValues.add(response.longValue);
        }
        if (response.decimalValue != null && response.decimalValues.isEmpty()) {
            response.decimalValues.add(response.decimalValue);
        }
        if (response.stringValue != null && response.stringValues.isEmpty()) {
            response.stringValues.add(response.stringValue);
        }
        if (response.booleanValue != null && response.booleanValues.isEmpty()) {
            response.booleanValues.add(response.booleanValue);
        }
        if (response.dateValue != null && response.dateValues.isEmpty()) {
            response.dateValues.add(response.dateValue);
        }
        if (response.dateTimeValue != null && response.dateTimeValues.isEmpty()) {
            response.dateTimeValues.add(response.dateTimeValue);
        }

        return response;
    }

    private static Long toLongValue( Decimal value ) {
        return ( value == null ? null : value.longValue() );
    }

    private static Long toLongValue( String value ) {
        return ( ( value == null || !value.isNumeric() ) ? null : Long.valueOf( value ) );
    }

    private static Long toLongValue( Boolean value ) {
        return ( value == null ? null : value ? 1 : 0 );
    }

    private static Long toLongValue( Date value ) {
        return ( value == null ? null : toLongValue(
            DateTime.newInstance( value.year(), value.month(), value.day(), 0, 0, 0 )
        ));
    }

    private static Long toLongValue( DateTime value ) {
        return ( value == null ? null : value.getTime() );
    }

    private static Decimal toDecimalValue( String value ) {
        return ( ( value == null || !value.isNumeric() ) ? null : Decimal.valueOf( value ) );
    }

    private static Decimal toDecimalValue( Boolean value ) {
        return ( value == null ? null : value ? 1 : 0 );
    }

    private static Decimal toDecimalValue( Date value ) {
        return ( value == null ? null : toDecimalValue(
            DateTime.newInstance( value.year(), value.month(), value.day(), 0, 0, 0 )
        ));
    }

    private static Decimal toDecimalValue( DateTime value ) {
        return ( value == null ? null : value.getTime() );
    }

    private static String toStringValue( Date value ) {
        return ( value == null ? null : String.valueOf( value ) );
    }

    private static String toStringValue( DateTime value ) {
        return ( value == null ? null : String.valueOf( value ) );
    }

    private static String toStringValue( Object value ) {
        return ( value == null ? null : String.valueOf( value ) );
    }

    private static Boolean toBooleanValue( Decimal value ) {
        return ( value == null ? false : ( value > 0 ) );
    }

    private static Boolean toBooleanValue( String value ) {
        return ( value == null ? false : Boolean.valueOf( value ) );
    }

    private static Boolean toBooleanValue( Date value ) {
        return ( value == null ? false : true );
    }

    private static Boolean toBooleanValue( DateTime value ) {
        return ( value == null ? false : true );
    }

    private static Date toDateValue( Decimal value ) {
        return ( value == null ? null : DateTime.newInstance( value.longValue() ).date() );
    }

    private static Date toDateValue( String value ) {
        try {
            return ( value == null ? null : Date.valueOf( value ) );
        } catch ( Exception e ) {
            return null;
        }
    }

    private static Date toDateValue( Boolean value ) {
        return null;
    }

    private static Date toDateValue( DateTime value ) {
        return ( value == null ? null : value.date() );
    }

    private static DateTime toDateTimeValue( Decimal value ) {
        return ( value == null ? null : DateTime.newInstance( value.longValue() ) );
    }

    private static DateTime toDateTimeValue( String value ) {
        try {
            return ( value == null ? null : DateTime.valueOf( value ) );
        } catch ( Exception e ) {
            return toDateValue( value );
        }
    }

    private static DateTime toDateTimeValue( Boolean value ) {
        return null;
    }

    private static DateTime toDateTimeValue( Date value ) {
        return ( value == null ? null : DateTime.newInstance( value.year(), value.month(), value.day(), 0, 0, 0 ) );
    }

    public class Request {

        @InvocableVariable(
            label = 'Parameter Name'
            required = true
        )
        public String paramName;

    }

    public class Response {

        @InvocableVariable(
            label = 'Number'
            description = 'Whole number (no decimals)'
        )
        public Long longValue;

        @InvocableVariable(
            label = 'Decimal'
            description = 'Number that may include decimal places'
        )
        public Decimal decimalValue;

        @InvocableVariable(
            label = 'Text'
            description = 'Text value'
        )
        public String stringValue;

        @InvocableVariable(
            label = 'Boolean'
            description = 'true/false'
        )
        public Boolean booleanValue;

        @InvocableVariable(
            label = 'Date'
        )
        public Date dateValue;

        @InvocableVariable(
            label = 'Date/Time'
        )
        public DateTime dateTimeValue;

        @InvocableVariable(
            label = 'Number Collection'
            description = 'Collection of whole numbers'
        )
        public Long[] longValues = new Long[] {};

        @InvocableVariable(
            label = 'Decimal Collection'
            description = 'Collection of numbers that may include decimal places'
        )
        public Decimal[] decimalValues = new Decimal[] {};

        @InvocableVariable(
            label = 'Text Collection'
            description = 'Collection of text values'
        )
        public String[] stringValues = new String[] {};

        @InvocableVariable(
            label = 'Boolean Collection'
            description = 'Collection of true/false values'
        )
        public Boolean[] booleanValues = new Boolean[] {};

        @InvocableVariable(
            label = 'Date Collection'
            description = 'Collection of date values'
        )
        public Date[] dateValues = new Date[] {};

        @InvocableVariable(
            label = 'Date/Time Collection'
            description = 'Collection of date/time values'
        )
        public DateTime[] dateTimeValues = new DateTime[] {};

    }

}